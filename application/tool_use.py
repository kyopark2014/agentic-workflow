import logging
import sys
import traceback
import json
import time
import boto3
import re
import requests
import datetime
import knowledge_base as kb
import chat
import functools
import utils

from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Literal
from langchain_core.tools import tool
from bs4 import BeautifulSoup
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_aws import AmazonKnowledgeBasesRetriever
from pytz import timezone
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.docstore.document import Document
from urllib import parse
from langgraph.graph import START, END, StateGraph

logger = utils.CreateLogger('tool_use')
try:
    with open("/home/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        logger.info(f"config: {config}")

except Exception:
    logger.info(f"use local configuration")
    with open("application/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        logger.info(f"config: {config}")

# variables
bedrock_region = config["region"] if "region" in config else "us-west-2"
projectName = config["projectName"] if "projectName" in config else "langgraph-nova"

path = config["sharing_url"] if "sharing_url" in config else None
if path is None:
    raise Exception ("No Sharing URL")

numberOfDocs = 4
s3_prefix = 'docs'
doc_prefix = s3_prefix+'/'

reference_docs = []
contentList = []
# api key to get weather information in agent
secretsmanager = boto3.client(
    service_name='secretsmanager',
    region_name=bedrock_region
)

try:
    get_weather_api_secret = secretsmanager.get_secret_value(
        SecretId=f"openweathermap-{projectName}"
    )
    #print('get_weather_api_secret: ', get_weather_api_secret)
    if get_weather_api_secret['SecretString']:
        secret = json.loads(get_weather_api_secret['SecretString'])
        #print('secret: ', secret)
        weather_api_key = secret['weather_api_key']
    else:
        logger.info(f"No secret found for weather api")

except Exception as e:
    raise e

# api key to use Tavily Search
tavily_key = tavily_api_wrapper = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    #print('secret: ', secret)

    if "tavily_api_key" in secret:
        tavily_key = secret['tavily_api_key']
        #print('tavily_api_key: ', tavily_api_key)

        if tavily_key:
            tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
            #     os.environ["TAVILY_API_KEY"] = tavily_key

            # Tavily Tool Test
            # query = 'what is Amazon Nova Pro?'
            # search = TavilySearchResults(
            #     max_results=1,
            #     include_answer=True,
            #     include_raw_content=True,
            #     api_wrapper=tavily_api_wrapper,
            #     search_depth="advanced", # "basic"
            #     # include_domains=["google.com", "naver.com"]
            # )
            # output = search.invoke(query)
            # print('tavily output: ', output)    
        else:
            logger.info(f"tavily_key is required.")
except Exception as e: 
    logger.info(f"Tavily credential is required: {e}")
    raise e

# funtions
def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        logger.info(f"Not Korean: {word_kor}")
        return False

def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags." 
        "Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")       
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def get_basic_answer(query):
    logger.info(f"#### get_basic_answer ####")
    llm = chat.get_chat()

    if isKorean(query)==True:
        system = (
            "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "You will be acting as a thoughtful advisor."
            "Using the following conversation, answer friendly for the newest question." 
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."     
        )    
    
    human = "Question: {input}"    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        ("human", human)
    ])    
    
    chain = prompt | llm    
    output = chain.invoke({"input": query})
    logger.info(f"output.content: {output.content}")

    return output.content

####################### LangGraph #######################
# Agentic Workflow: Tool Use
#########################################################

@tool 
def get_book_list(keyword: str) -> str:
    """
    Search book list by keyword and then return book list
    keyword: search keyword
    return: book list
    """
    
    keyword = keyword.replace('\'','')

    answer = ""
    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        
        if len(prod_info):
            answer = "추천 도서는 아래와 같습니다.\n"
            
        for prod in prod_info[:5]:
            title = prod.text.strip().replace("\n", "")       
            link = prod.get("href")
            answer = answer + f"{title}, URL: {link}\n\n"
    
    return answer

@tool
def get_current_time(format: str=f"%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in the specified format"""
    # f"%Y-%m-%d %H:%M:%S"
    
    format = format.replace('\'','')
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    logger.info(f"timestr: {timestr}")
    
    return timestr

@tool
def get_weather_info(city: str) -> str:
    """
    retrieve weather information by city name and then return weather statement.
    city: the name of city to retrieve
    return: weather statement
    """    
    
    city = city.replace('\n','')
    city = city.replace('\'','')
    city = city.replace('\"','')
                
    llm = chat.get_chat()
    if isKorean(city):
        place = traslation(llm, city, "Korean", "English")
        logger.info(f"city (translated): {place}")
    else:
        place = city
        city = traslation(llm, city, "English", "Korean")
        logger.info(f"city (translated): {city}")
        
    logger.info(f"place: {place}")
    
    weather_str: str = f"{city}에 대한 날씨 정보가 없습니다."
    if weather_api_key: 
        apiKey = weather_api_key
        lang = 'en' 
        units = 'metric' 
        api = f"https://api.openweathermap.org/data/2.5/weather?q={place}&APPID={apiKey}&lang={lang}&units={units}"
        # print('api: ', api)
                
        try:
            result = requests.get(api)
            result = json.loads(result.text)
            logger.info(f"result: {result}")
        
            if 'weather' in result:
                overall = result['weather'][0]['main']
                current_temp = result['main']['temp']
                min_temp = result['main']['temp_min']
                max_temp = result['main']['temp_max']
                humidity = result['main']['humidity']
                wind_speed = result['wind']['speed']
                cloud = result['clouds']['all']
                
                #weather_str = f"{city}의 현재 날씨의 특징은 {overall}이며, 현재 온도는 {current_temp}도 이고, 최저온도는 {min_temp}도, 최고 온도는 {max_temp}도 입니다. 현재 습도는 {humidity}% 이고, 바람은 초당 {wind_speed} 미터 입니다. 구름은 {cloud}% 입니다."
                weather_str = f"{city}의 현재 날씨의 특징은 {overall}이며, 현재 온도는 {current_temp} 입니다. 현재 습도는 {humidity}% 이고, 바람은 초당 {wind_speed} 미터 입니다. 구름은 {cloud}% 입니다."
                #weather_str = f"Today, the overall of {city} is {overall}, current temperature is {current_temp} degree, min temperature is {min_temp} degree, highest temperature is {max_temp} degree. huminity is {humidity}%, wind status is {wind_speed} meter per second. the amount of cloud is {cloud}%."            
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")   
            # raise Exception ("Not able to request to LLM")    
        
    logger.info(f"weather_str: {weather_str}")                        
    return weather_str

# Tavily Tool
tavily_tool = TavilySearchResults(
    max_results=3,
    include_answer=True,
    include_raw_content=True,
    api_wrapper=tavily_api_wrapper,
    search_depth="advanced", # "basic"
    # include_domains=["google.com", "naver.com"]
)
     
@tool    
def search_by_knowledge_base(keyword: str) -> str:
    """
    Search technical information by keyword and then return the result as a string.
    keyword: search keyword
    return: the technical information of keyword
    """    
    logger.info(f"###### search_by_knowledge_base ######") 
    
    global reference_docs
 
    logger.info(f"keyword: {keyword}")
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    logger.info(f"modified keyword: {keyword}")
    
    top_k = numberOfDocs
    relevant_docs = []
    if kb.knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=kb.knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID"   # SEMANTIC
            }},
        )
        
        docs = retriever.invoke(keyword)
        # print('docs: ', docs)
        logger.info(f"--> docs from knowledge base")
        for i, doc in enumerate(docs):
            # print_doc(i, doc)
            
            content = ""
            if doc.page_content:
                content = doc.page_content
            
            score = doc.metadata["score"]
            
            link = ""
            if "s3Location" in doc.metadata["location"]:
                link = doc.metadata["location"]["s3Location"]["uri"] if doc.metadata["location"]["s3Location"]["uri"] is not None else ""
                
                # print('link:', link)    
                pos = link.find(f"/{doc_prefix}")
                name = link[pos+len(doc_prefix)+1:]
                encoded_name = parse.quote(name)
                # print('name:', name)
                link = f"{path}/{doc_prefix}{encoded_name}"
                
            elif "webLocation" in doc.metadata["location"]:
                link = doc.metadata["location"]["webLocation"]["url"] if doc.metadata["location"]["webLocation"]["url"] is not None else ""
                name = "WEB"

            url = link
            # print('url:', url)
            
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'score': score,
                        'url': url,
                        'from': 'RAG'
                    },
                )
            )    
    
    # grading        
    filtered_docs = chat.grade_documents(keyword, relevant_docs)

    filtered_docs = chat.check_duplication(filtered_docs) # duplication checker

    relevant_context = ""
    for i, document in enumerate(filtered_docs):
        logger.info(f"{i}: {document}")
        if document.page_content:
            relevant_context += document.page_content + "\n\n"        
    logger.info(f"relevant_context: {relevant_context}")
    
    if len(filtered_docs):
        reference_docs += filtered_docs
        return relevant_context
    else:        
        # relevant_context = "No relevant documents found."
        relevant_context = "관련된 정보를 찾지 못하였습니다."
        logger.info(f"--> {relevant_context}")
        return relevant_context
    
@tool
def search_by_tavily(keyword: str) -> str:
    """
    Search general knowledge by keyword and then return the result as a string.
    keyword: search keyword
    return: the information of keyword
    """    
    global reference_docs    
    answer = ""
    
    if tavily_key:
        keyword = keyword.replace('\'','')
        
        search = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=True,
            api_wrapper=tavily_api_wrapper,
            search_depth="advanced", # "basic"
            # include_domains=["google.com", "naver.com"]
        )
                    
        try: 
            output = search.invoke(keyword)
            if output[:9] == "HTTPError":
                logger.info(f"output: {output}")
                raise Exception ("Not able to request to tavily")
            else:        
                logger.info(f"tavily outpu: {output}")
                if output == "HTTPError('429 Client Error: Too Many Requests for url: https://api.tavily.com/search')":            
                    raise Exception ("Not able to request to tavily")
                
                for result in output:
                    logger.info(f"result: {result}")
                    if result:
                        content = result.get("content")
                        url = result.get("url")
                        
                        reference_docs.append(
                            Document(
                                page_content=content,
                                metadata={
                                    'name': 'WWW',
                                    'url': url,
                                    'from': 'tavily'
                                },
                            )
                        )                
                        answer = answer + f"{content}, URL: {url}\n"        
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")           
            # raise Exception ("Not able to request to tavily")  

    if answer == "":
        # answer = "No relevant documents found." 
        answer = "관련된 정보를 찾지 못하였습니다."
                     
    return answer

@tool
def stock_data_lookup(ticker, country):
    """
    Retrieve accurate stock trends for a given ticker.
    ticker: the ticker to retrieve price history for
    country: the english country name of the stock
    return: the information of ticker
    """ 
    com = re.compile('[a-zA-Z]') 
    alphabet = com.findall(ticker)
    logger.info(f"alphabet: {alphabet}")

    logger.info(f"country: {country}")

    if len(alphabet)==0:
        if country == "South Korea":
            ticker += ".KS"
        elif country == "Japan":
            ticker += ".T"
    logger.info(f"ticker: {ticker}")
    
    stock = yf.Ticker(ticker)
    
    # get the price history for past 1 month
    history = stock.history(period="1mo")
    logger.info(f"history: {history}")
    
    result = f"## Trading History\n{history}"
    #history.reset_index().to_json(orient="split", index=False, date_format="iso")    
    
    result += f"\n\n## Financials\n{stock.financials}"    
    logger.info(f"financials: {stock.financials}")

    result += f"\n\n## Major Holders\n{stock.major_holders}"
    logger.info(f"major_holders: {stock.major_holders}")

    logger.info(f"result: {result}")

    return result

tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_knowledge_base, stock_data_lookup]        

####################### LangGraph #######################
# Chat Agent Executor
#########################################################
def run_agent_executor(query, st):
    chatModel = chat.get_chat()     
    model = chatModel.bind_tools(tools)

    class State(TypedDict):
        # messages: Annotated[Sequence[BaseMessage], operator.add]
        messages: Annotated[list, add_messages]

    tool_node = ToolNode(tools)

    def should_continue(state: State) -> Literal["continue", "end"]:
        logger.info(f"###### should_continue ######")

        logger.info(f"state: {state}")
        messages = state["messages"]    

        last_message = messages[-1]
        logger.info(f"last_message: {last_message}")
        
        # print('last_message: ', last_message)
        
        # if not isinstance(last_message, ToolMessage):
        #     return "end"
        # else:                
        #     return "continue"
        if isinstance(last_message, ToolMessage) or last_message.tool_calls:
            logger.info(f"tool_calls: {last_message.tool_calls}")

            for message in last_message.tool_calls:
                logger.info(f"tool name: {message['name']}, args: {message['args']}")
                # update_state_message(f"calling... {message['name']}", config)

            logger.info(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
            return "continue"
        
        #if not last_message.tool_calls:
        else:
            logger.info(f"Final: {last_message.content}")
            print("--- END ---")
            logger.info(f"--- END ---")
            return "end"
           
    def call_model(state: State, config):
        logger.info(f"###### call_model ######")
        logger.info(f"state: {state['messages']}")
                
        if isKorean(state["messages"][0].content)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
                "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "한국어로 답변하세요."
            )
        else: 
            system = (            
                "You are a conversational AI designed to answer in a friendly way to a question."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            )

        for attempt in range(3):   
            logger.info(f"attempt: {attempt}")
            try:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )
                chain = prompt | model
                    
                response = chain.invoke(state["messages"])
                logger.info(f"call_model response: {response}")

                if isinstance(response.content, list):            
                    for re in response.content:
                        if "type" in re:
                            if re['type'] == 'text':
                                logger.info(f"--> {re['type']}: {re['text']}")

                                status = re['text']
                                logger.info(f"status: {status}")
                                
                                status = status.replace('`','')
                                status = status.replace('\"','')
                                status = status.replace("\'",'')
                                
                                logger.info(f"status: {status}")
                                if status.find('<thinking>') != -1:
                                    logger.info(f"Remove <thinking> tag.")
                                    status = status[status.find('<thinking>')+11:status.find('</thinking>')]
                                    logger.info(f"status without tag: {status}")

                                if chat.debug_mode=="Enable":
                                    st.info(status)
                                
                            elif re['type'] == 'tool_use':                
                                logger.info(f"--> {re['type']}: {re['name']}, {re['input']}")

                                if chat.debug_mode=="Enable":
                                    st.info(f"{re['type']}: {re['name']}, {re['input']}")
                            else:
                                logger.info(re)
                        else: # answer
                            logger.info(response.content)
                break
            except Exception:
                response = AIMessage(content="답변을 찾지 못하였습니다.")

                err_msg = traceback.format_exc()
                logger.info(f"error message: {err_msg}")
                # raise Exception ("Not able to request to LLM")

        return {"messages": [response]}

    def buildChatAgent():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")

        return workflow.compile()

    # initiate
    global reference_docs, contentList
    reference_docs = []
    contentList = []

    # workflow 
    app = buildChatAgent()
            
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50
    }
    
    # msg = message.content
    result = app.invoke({"messages": inputs}, config)
    #print("result: ", result)

    msg = result["messages"][-1].content
    logger.info(f"msg: {msg}")

    for i, doc in enumerate(reference_docs):
        logger.info(f"--> {i}: {doc}")
        
    reference = ""
    if reference_docs:
        reference = chat.get_references(reference_docs)

    msg = chat.extract_thinking_tag(msg, st)
    
    return msg+reference, reference_docs

####################### LangGraph #######################
# Agentic Workflow: Tool Use (partial tool을 활용)
#########################################################

def run_agent_executor2(query, st):        
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        answer: str

    tool_node = ToolNode(tools)
            
    def create_agent(llm, tools):        
        tool_names = ", ".join([tool.name for tool in tools])
        logger.info(f"tool_names: {tool_names}")

        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."

            "Use the provided tools to progress towards answering the question."
            "If you are unable to fully answer, that's OK, another assistant with different tools "
            "will help where you left off. Execute what you can to make progress."
            "You have access to the following tools: {tool_names}."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        prompt = prompt.partial(tool_names=tool_names)
        
        return prompt | llm.bind_tools(tools)
    
    def agent_node(state, agent, name):
        logger.info(f"###### agent_node:{name} ######")

        last_message = state["messages"][-1]
        logger.info(f"last_message: {last_message}")
        if isinstance(last_message, ToolMessage) and last_message.content=="":    
            logger.info(f"last_message is empty")
            answer = get_basic_answer(state["messages"][0].content)  
            return {
                "messages": [AIMessage(content=answer)],
                "answer": answer
            }
        
        response = agent.invoke(state["messages"])
        logger.info(f"response: {response}")

        if "answer" in state:
            answer = state['answer']
        else:
            answer = ""

        if isinstance(response.content, list):      
            for re in response.content:
                if "type" in re:
                    if re['type'] == 'text':
                        logger.info(f"--> {re['type']}: {re['text']}")

                        status = re['text']
                        if status.find('<thinking>') != -1:
                            logger.info(f"Remove <thinking> tag.")
                            status = status[status.find('<thinking>')+11:status.find('</thinking>')]
                            logger.info(f"'status without tag: {status}")

                        if chat.debug_mode=="Enable":
                            st.info(status)

                    elif re['type'] == 'tool_use':                
                        logger.info(f"--> {re['type']}: name: {re['name']}, input: {re['input']}")

                        if chat.debug_mode=="Enable":
                            st.info(f"{re['type']}: name: {re['name']}, input: {re['input']}")
                    else:
                        logger.info(re)
                else: # answer
                    answer += '\n'+response.content
                    logger.info(response.content)
                    break

        response = AIMessage(**response.dict(exclude={"type", "name"}), name=name)     
        logger.info(f"message: {response}")
        
        return {
            "messages": [response],
            "answer": answer
        }
    
    def final_answer(state):
        logger.info(f"###### final_answer ######")  

        answer = ""        
        if "answer" in state:
            answer = state['answer']            
        else:
            answer = state["messages"][-1].content

        if answer.find('<thinking>') != -1:
            logger.info(f"Remove <thinking> tag.")
            answer = answer[answer.find('</thinking>')+12:]
        logger.info(f"answer: {answer}")
        
        return {
            "answer": answer
        }
    
    llm = chat.get_chat()
    
    execution_agent = create_agent(llm, tools)
    
    execution_agent_node = functools.partial(agent_node, agent=execution_agent, name="execution_agent")
    
    def should_continue(state: State, config) -> Literal["continue", "end"]:
        logger.info(f"###### should_continue ######")
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
        
        last_message = messages[-1]        
        if not last_message.tool_calls:
            logger.info(f"Final: {last_message.content}")
            logger.info(f"-- END ---")
            return "end"
        else:      
            logger.info(f"tool_calls: {last_message.tool_calls}")

            for message in last_message.tool_calls:
                logger.info(f"tool name: {message['name']}, args: {message['args']}")
                # update_state_message(f"calling... {message['name']}", config)

            logger.info(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
            return "continue"

    def buildAgentExecutor():
        workflow = StateGraph(State)

        workflow.add_node("agent", execution_agent_node)
        workflow.add_node("action", tool_node)
        workflow.add_node("final_answer", final_answer)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": "final_answer",
            },
        )
        workflow.add_edge("action", "agent")
        workflow.add_edge("final_answer", END)

        return workflow.compile()

    app = buildAgentExecutor()
            
    inputs = [HumanMessage(content=query)]
    config = {"recursion_limit": 50}
    
    msg = ""
    # for event in app.stream({"messages": inputs}, config, stream_mode="values"):   
    #     # print('event: ', event)
        
    #     if "answer" in event:
    #         msg = event["answer"]
    #     else:
    #         msg = event["messages"][-1].content
    #     # print('message: ', message)

    output = app.invoke({"messages": inputs}, config)
    logger.info(f"output: {output}")

    msg = output['answer']

    return msg, reference_docs
