import traceback
import boto3
import os
import json
import re
import requests
import datetime
import functools
import uuid
import time
import logging
import base64
import operator
import info
import PyPDF2
import csv

from io import BytesIO
from PIL import Image
from pytz import timezone
from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import tool
from langchain.docstore.document import Document
from tavily import TavilyClient  
from langchain_community.tools.tavily_search import TavilySearchResults
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from typing import Any, List, Tuple, Dict, Optional, cast, Literal, Sequence, Union
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_aws import AmazonKnowledgeBasesRetriever
from multiprocessing import Process, Pipe
from urllib import parse
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    with open("/home/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        print('config: ', config)
except Exception:
    print("use local configuration")
    with open("application/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        print('config: ', config)

bedrock_region = config["region"] if "region" in config else "us-west-2"

projectName = config["projectName"] if "projectName" in config else "langgraph-nova"

accountId = config["accountId"] if "accountId" in config else None
if accountId is None:
    raise Exception ("No accountId")

region = config["region"] if "region" in config else "us-west-2"
print('region: ', region)

s3_prefix = 'docs'

knowledge_base_role = config["knowledge_base_role"] if "knowledge_base_role" in config else None
if knowledge_base_role is None:
    raise Exception ("No Knowledge Base Role")

collectionArn = config["collectionArn"] if "collectionArn" in config else None
if collectionArn is None:
    raise Exception ("No collectionArn")

vectorIndexName = projectName

opensearch_url = config["opensearch_url"] if "opensearch_url" in config else None
if opensearch_url is None:
    raise Exception ("No OpenSearch URL")

path = config["sharing_url"] if "sharing_url" in config else None
if path is None:
    raise Exception ("No Sharing URL")

credentials = boto3.Session().get_credentials()
service = "aoss" 
awsauth = AWSV4SignerAuth(credentials, region, service)

s3_arn = config["s3_arn"] if "s3_arn" in config else None
if s3_arn is None:
    raise Exception ("No S3 ARN")

s3_bucket = config["s3_bucket"] if "s3_bucket" in config else None
if s3_bucket is None:
    raise Exception ("No storage!")

parsingModelArn = f"arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
embeddingModelArn = f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"

knowledge_base_name = projectName

numberOfDocs = 4
MSG_LENGTH = 100    
grade_state = "LLM" # LLM, OTHERS

doc_prefix = s3_prefix+'/'
useEnhancedSearch = False

userId = "demo"
map_chain = dict() 

model_name = "Nova Pro"
model_type = "nova"
multi_region = 'Enable'
debug_mode = "Enable"

models = info.get_model_info(model_name)
number_of_models = len(models)
selected_chat = 0

def update(modelName, debugMode, multiRegion):    
    global model_name, debug_mode, multi_region     
    global selected_chat, models, number_of_models
    
    if model_name != modelName:
        model_name = modelName
        print('model_name: ', model_name)
        
        selected_chat = 0
        models = info.get_model_info(model_name)
        number_of_models = len(models)
        
    if debug_mode != debugMode:
        debug_mode = debugMode
        print('debug_mode: ', debug_mode)

    if multi_region != multiRegion:
        multi_region = multiRegion
        print('multi_region: ', multi_region)
        
        selected_chat = 0
        
def initiate():
    global userId
    global memory_chain

    userId = uuid.uuid4().hex
    print('userId: ', userId)

    if userId in map_chain:  
            # print('memory exist. reuse it!')
            memory_chain = map_chain[userId]
    else: 
        # print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

initiate()

def clear_chat_history():
    memory_chain = []
    map_chain[userId] = memory_chain

def save_chat_history(text, msg):
    memory_chain.chat_memory.add_user_message(text)
    if len(msg) > MSG_LENGTH:
        memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
    else:
        memory_chain.chat_memory.add_ai_message(msg) 

def get_chat():
    global selected_chat, model_type

    profile = models[selected_chat]
    # print('profile: ', profile)
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k
    print(f'LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}')

    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [STOP_SEQUENCE]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=bedrock_region
    )    
    
    if multi_region=='Enable':
        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    else:
        selected_chat = 0

    return chat

def get_parallel_processing_chat(models, selected):
    global model_type
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    maxOutputTokens = 4096
    print(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}')

    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [STOP_SEQUENCE]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )        
    return chat

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"{i}: {text}, metadata:{doc.metadata}")

reference_docs = []
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
        print('No secret found for weather api')

except Exception as e:
    raise e

# api key to use LangSmith
langsmith_api_key = ""
try:
    get_langsmith_api_secret = secretsmanager.get_secret_value(
        SecretId=f"langsmithapikey-{projectName}"
    )
    #print('get_langsmith_api_secret: ', get_langsmith_api_secret)
    if get_langsmith_api_secret['SecretString']:
        secret = json.loads(get_langsmith_api_secret['SecretString'])
        #print('secret: ', secret)
        langsmith_api_key = secret['langsmith_api_key']
        langchain_project = secret['langchain_project']
    else:
        print('No secret found for lengsmith api')
except Exception as e:
    raise e

if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

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
            print('tavily_key is required.')
except Exception as e: 
    print('Tavily credential is required: ', e)
    raise e

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
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
        print('error message: ', err_msg)          
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def upload_to_s3(file_bytes, file_name):
    """
    Upload a file to S3 and return the URL
    """
    try:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )
        # Generate a unique file name to avoid collisions
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #unique_id = str(uuid.uuid4())[:8]
        #s3_key = f"uploaded_images/{timestamp}_{unique_id}_{file_name}"
        s3_key = f"{s3_prefix}/{file_name}"

        if file_name.lower().endswith((".jpg", ".jpeg")):
            content_type = "image/jpeg"
        elif file_name.lower().endswith((".pdf")):
            content_type = "application/pdf"
        elif file_name.lower().endswith((".txt")):
            content_type = "text/plain"
        elif file_name.lower().endswith((".csv")):
            content_type = "text/csv"
        elif file_name.lower().endswith((".ppt", ".pptx")):
            content_type = "application/vnd.ms-powerpoint"
        elif file_name.lower().endswith((".doc", ".docx")):
            content_type = "application/msword"
        elif file_name.lower().endswith((".xls")):
            content_type = "application/vnd.ms-excel"
        elif file_name.lower().endswith((".py")):
            content_type = "text/x-python"
        elif file_name.lower().endswith((".js")):
            content_type = "application/javascript"
        elif file_name.lower().endswith((".md")):
            content_type = "text/markdown"
        elif file_name.lower().endswith((".png")):
            content_type = "image/png"
        elif file_name.lower().endswith((".json")):
            content_type = "application/json"
        
        user_meta = {  # user-defined metadata
            "content_type": content_type,
            "model_name": model_name,
            "multi_region": multi_region
        }
        
        response = s3_client.put_object(
            Bucket=s3_bucket, 
            Key=s3_key, 
            ContentType=content_type,
            Metadata = user_meta,
            Body=file_bytes            
        )
        print('upload response: ', response)

        url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
        return url
    
    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        print(err_msg)
        return None

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_parallel_processing_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    # print(f"score: {score}")
    
    grade = score.binary_score    
    if grade == 'yes':
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()

def grade_documents_using_parallel_processing(question, documents):
    global selected_chat
    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, models, selected_chat))
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        relevant_doc = parent_conn.recv()

        if relevant_doc is not None:
            filtered_docs.append(relevant_doc)

    for process in processes:
        process.join()
    
    return filtered_docs

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_retrieval_grader(chat):
    system = (
        "You are a grader assessing relevance of a retrieved document to a user question."
        "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    print("start grading...")
    print("grade_state: ", grade_state)
    
    if grade_state == "LLM":
        filtered_docs = []
        if multi_region == 'Enable':  # parallel processing        
            filtered_docs = grade_documents_using_parallel_processing(question, documents)

        else:
            # Score each doc    
            chat = get_chat()
            retrieval_grader = get_retrieval_grader(chat)
            for i, doc in enumerate(documents):
                # print('doc: ', doc)
                print_doc(i, doc)
                
                score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                # print("score: ", score)
                
                grade = score.binary_score
                # print("grade: ", grade)
                # Document relevant
                if grade.lower() == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(doc)
                # Document not relevant
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    # We do not include the document in filtered_docs
                    # We set a flag to indicate that we want to run web search
                    continue
    
    else:  # OTHERS
        filtered_docs = documents

    return filtered_docs

contentList = []
def check_duplication(docs):
    global contentList
    length_original = len(docs)
    
    updated_docs = []
    print('length of relevant_docs:', len(docs))
    for doc in docs:            
        if doc.page_content in contentList:
            print('duplicated!')
            continue
        contentList.append(doc.page_content)
        updated_docs.append(doc)            
    length_updated_docs = len(updated_docs)   
    
    if length_original == length_updated_docs:
        print('no duplication')
    else:
        print('length of updated relevant_docs: ', length_updated_docs)
    
    return updated_docs

def retrieve_documents_from_tavily(query, top_k):
    print("###### retrieve_documents_from_tavily ######")

    relevant_documents = []        
    search = TavilySearchResults(
        max_results=top_k,
        include_answer=True,
        include_raw_content=True,        
        api_wrapper=tavily_api_wrapper,
        search_depth="advanced", 
        # include_domains=["google.com", "naver.com"]
    )
                    
    try: 
        output = search.invoke(query)
        print('tavily output: ', output)

        if output[:9] == "HTTPError":
            print('output: ', output)
            raise Exception ("Not able to request to tavily")
        else:        
            print(f"--> tavily query: {query}")
            for i, result in enumerate(output):
                print(f"{i}: {result}")
                if result:
                    content = result.get("content")
                    url = result.get("url")
                    
                    relevant_documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                'name': 'WWW',
                                'url': url,
                                'from': 'tavily'
                            },
                        )
                    )                   
    
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        # raise Exception ("Not able to request to tavily")   

    return relevant_documents 

def get_references(docs):    
    reference = ""
    for i, doc in enumerate(docs):
        page = ""
        if "page" in doc.metadata:
            page = doc.metadata['page']
            #print('page: ', page)            
        url = ""
        if "url" in doc.metadata:
            url = doc.metadata['url']
            print('url: ', url)
        name = ""
        if "name" in doc.metadata:
            name = doc.metadata['name']
            #print('name: ', name)     
        
        sourceType = ""
        if "from" in doc.metadata:
            sourceType = doc.metadata['from']
        else:
            # if useEnhancedSearch:
            #     sourceType = "OpenSearch"
            # else:
            #     sourceType = "WWW"
            sourceType = "WWW"

        #print('sourceType: ', sourceType)        
        
        #if len(doc.page_content)>=1000:
        #    excerpt = ""+doc.page_content[:1000]
        #else:
        #    excerpt = ""+doc.page_content
        excerpt = ""+doc.page_content
        # print('excerpt: ', excerpt)
        
        # for some of unusual case 
        #excerpt = excerpt.replace('"', '')        
        #excerpt = ''.join(c for c in excerpt if c not in '"')
        excerpt = re.sub('"', '', excerpt)
        excerpt = re.sub('#', '', excerpt)        
        print('excerpt(quotation removed): ', excerpt)
        
        if page:                
            reference += f"{i+1}. {page}page in [{name}]({url})), {excerpt[:30]}...\n"
        else:
            reference += f"{i+1}. [{name}]({url}), {excerpt[:30]}...\n"

    if reference: 
        reference = "\n\n#### 관련 문서\n"+reference

    return reference

def tavily_search(query, k):
    docs = []    
    try:
        tavily_client = TavilyClient(
            api_key=tavily_key
        )
        response = tavily_client.search(query, max_results=k)
        # print('tavily response: ', response)
            
        for r in response["results"]:
            name = r.get("title")
            if name is None:
                name = 'WWW'
            
            docs.append(
                Document(
                    page_content=r.get("content"),
                    metadata={
                        'name': name,
                        'url': r.get("url"),
                        'from': 'tavily'
                    },
                )
            )                   
    except Exception as e:
        print('Exception: ', e)

    return docs

def extract_thinking_tag(response, st):
    if response.find('<thinking>') != -1:
        status = response[response.find('<thinking>')+11:response.find('</thinking>')]
        print('agent_thinking: ', status)
        
        if debug_mode=="Enable":
            st.info(status)

        if response.find('<thinking>') == 0:
            msg = response[response.find('</thinking>')+13:]
        else:
            msg = response[:response.find('<thinking>')]
        print('msg: ', msg)
    else:
        msg = response

    return msg

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    key = s3_prefix+'/'+s3_file_name
    print(f"bucket: {s3_bucket}, key: {key}")
    doc = s3r.Object(s3_bucket, key)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(docs):    
    chat = get_chat()

    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        print('result of summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    contents = ""
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt' or file_type == 'md':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 
    texts = text_splitter.split_text(new_contents) 
    if texts:
        print('texts[0]: ', texts[0])
    
    return texts

def summary_of_code(code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다." 
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chat = get_chat()

    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "code": code
            }
        )
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def summary_image(img_base64, instruction):      
    chat = get_chat()

    if instruction:
        print('instruction: ', instruction)  
        query = f"{instruction}. <result> tag를 붙여주세요."
    else:
        query = "이미지가 의미하는 내용을 풀어서 자세히 알려주세요. markdown 포맷으로 답변을 작성합니다."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    for attempt in range(5):
        print('attempt: ', attempt)
        try: 
            result = chat.invoke(messages)
            
            extracted_text = result.content
            # print('summary from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            raise Exception ("Not able to request to LLM")
        
    return extracted_text

def extract_text(img_base64):    
    multimodal = get_chat()
    query = "텍스트를 추출해서 markdown 포맷으로 변환하세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    for attempt in range(5):
        print('attempt: ', attempt)
        try: 
            result = multimodal.invoke(messages)
            
            extracted_text = result.content
            # print('result of text extraction from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            raise Exception ("Not able to request to LLM")
    
    print('extracted_text: ', extracted_text)
    if len(extracted_text)<10:
        extracted_text = "텍스트를 추출하지 못하였습니다."    

    return extracted_text

fileId = uuid.uuid4().hex
# print('fileId: ', fileId)
def get_summary_of_uploaded_file(file_name, st):
    file_type = file_name[file_name.rfind('.')+1:len(file_name)]            
    print('file_type: ', file_type)
    
    if file_type == 'csv' or file_type == 'json':
        docs = load_csv_document(file_name)
        contexts = []
        for doc in docs:
            contexts.append(doc.page_content)
        print('contexts: ', contexts)
    
        msg = get_summary(contexts)

    elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
        texts = load_document(file_type, file_name)

        if len(texts):
            docs = []
            for i in range(len(texts)):
                docs.append(
                    Document(
                        page_content=texts[i],
                        metadata={
                            'name': file_name,
                            # 'page':i+1,
                            'url': path+'/'+doc_prefix+parse.quote(file_name)
                        }
                    )
                )
            print('docs[0]: ', docs[0])    
            print('docs size: ', len(docs))

            contexts = []
            for doc in docs:
                contexts.append(doc.page_content)
            print('contexts: ', contexts)

            msg = get_summary(contexts)
        else:
            msg = "문서 로딩에 실패하였습니다."
        
    elif file_type == 'py' or file_type == 'js':
        s3r = boto3.resource("s3")
        doc = s3r.Object(s3_bucket, s3_prefix+'/'+file_name)
        
        contents = doc.get()['Body'].read().decode('utf-8')
        
        #contents = load_code(file_type, object)                
                        
        msg = summary_of_code(contents, file_type)                  
        
    elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
        print('multimodal: ', file_name)
        
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )             
        if debug_mode=="Enable":
            status = "이미지를 가져옵니다."
            print('status: ', status)
            st.info(status)
            
        image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+file_name)
        # print('image_obj: ', image_obj)
        
        image_content = image_obj['Body'].read()
        img = Image.open(BytesIO(image_content))
        
        width, height = img.size 
        print(f"width: {width}, height: {height}, size: {width*height}")
        
        isResized = False
        while(width*height > 5242880):                    
            width = int(width/2)
            height = int(height/2)
            isResized = True
            print(f"width: {width}, height: {height}, size: {width*height}")
        
        if isResized:
            img = img.resize((width, height))
        
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
               
        # extract text from the image
        if debug_mode=="Enable":
            status = "이미지에서 텍스트를 추출합니다."
            print('status: ', status)
            st.info(status)
        
        text = extract_text(img_base64)
        # print('extracted text: ', text)

        if text.find('<result>') != -1:
            extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
            # print('extracted_text: ', extracted_text)
        else:
            extracted_text = text

        if debug_mode=="Enable":
            status = f"### 추출된 텍스트\n\n{extracted_text}"
            print('status: ', status)
            st.info(status)
    
        if debug_mode=="Enable":
            status = "이미지의 내용을 분석합니다."
            print('status: ', status)
            st.info(status)

        image_summary = summary_image(img_base64, "")
        print('image summary: ', image_summary)
            
        if len(extracted_text) > 10:
            contents = f"## 이미지 분석\n\n{image_summary}\n\n## 추출된 텍스트\n\n{extracted_text}"
        else:
            contents = f"## 이미지 분석\n\n{image_summary}"
        print('image contents: ', contents)

        msg = contents

    global fileId
    fileId = uuid.uuid4().hex
    # print('fileId: ', fileId)

    return msg

####################### LangChain #######################
# General Conversation
#########################################################

def general_conversation(query):
    chat = get_chat()

    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )
    
    human = "Question: {input}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)
    ])
                
    history = memory_chain.load_memory_variables({})["chat_history"]

    chain = prompt | chat | StrOutputParser()
    try: 
        stream = chain.stream(
            {
                "history": history,
                "input": query,
            }
        )  
            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to request to LLM: "+err_msg)
        
    return stream

    
####################### LangGraph #######################
# RAG: Knowledge Base
#########################################################

os_client = OpenSearch(
    hosts = [{
        'host': opensearch_url.replace("https://", ""), 
        'port': 443
    }],
    http_auth=awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class=RequestsHttpConnection,
)

def is_not_exist(index_name):    
    print('index_name: ', index_name)
        
    if os_client.indices.exists(index_name):
        print('use exist index: ', index_name)    
        return False
    else:
        print('no index: ', index_name)
        return True
    
knowledge_base_id = ""
data_source_id = ""
def initiate_knowledge_base():
    global knowledge_base_id, data_source_id
    #########################
    # opensearch index
    #########################
    if(is_not_exist(vectorIndexName)):
        print(f"creating opensearch index... {vectorIndexName}")        
        body={ 
            'settings':{
                "index.knn": True,
                "index.knn.algo_param.ef_search": 512,
                'analysis': {
                    'analyzer': {
                        'my_analyzer': {
                            'char_filter': ['html_strip'], 
                            'tokenizer': 'nori',
                            'filter': ['nori_number','lowercase','trim','my_nori_part_of_speech'],
                            'type': 'custom'
                        }
                    },
                    'tokenizer': {
                        'nori': {
                            'decompound_mode': 'mixed',
                            'discard_punctuation': 'true',
                            'type': 'nori_tokenizer'
                        }
                    },
                    "filter": {
                        "my_nori_part_of_speech": {
                            "type": "nori_part_of_speech",
                            "stoptags": [
                                    "E", "IC", "J", "MAG", "MAJ",
                                    "MM", "SP", "SSC", "SSO", "SC",
                                    "SE", "XPN", "XSA", "XSN", "XSV",
                                    "UNA", "NA", "VSV"
                            ]
                        }
                    }
                },
            },
            'mappings': {
                'properties': {
                    'vector_field': {
                        'type': 'knn_vector',
                        'dimension': 1024,
                        'method': {
                            "name": "hnsw",
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16
                            }
                        }                  
                    },
                    "AMAZON_BEDROCK_METADATA": {"type": "text", "index": False},
                    "AMAZON_BEDROCK_TEXT": {"type": "text"},
                }
            }
        }

        try: # create index
            response = os_client.indices.create(
                vectorIndexName,
                body=body
            )
            print('opensearch index was created:', response)

            # delay 5 seconds
            time.sleep(5)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                
            #raise Exception ("Not able to create the index")
            
    #########################
    # knowledge base
    #########################
    print('knowledge_base_name: ', knowledge_base_name)
    print('collectionArn: ', collectionArn)
    print('vectorIndexName: ', vectorIndexName)
    print('embeddingModelArn: ', embeddingModelArn)
    print('knowledge_base_role: ', knowledge_base_role)
    try: 
        client = boto3.client(
            service_name='bedrock-agent',
            region_name=bedrock_region
        )   
        response = client.list_knowledge_bases(
            maxResults=10
        )
        print('(list_knowledge_bases) response: ', response)
        
        if "knowledgeBaseSummaries" in response:
            summaries = response["knowledgeBaseSummaries"]
            for summary in summaries:
                if summary["name"] == knowledge_base_name:
                    knowledge_base_id = summary["knowledgeBaseId"]
                    print('knowledge_base_id: ', knowledge_base_id)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)
                    
    if not knowledge_base_id:
        print('creating knowledge base...')        
        for atempt in range(20):
            try:
                response = client.create_knowledge_base(
                    name=knowledge_base_name,
                    description="Knowledge base based on OpenSearch",
                    roleArn=knowledge_base_role,
                    knowledgeBaseConfiguration={
                        'type': 'VECTOR',
                        'vectorKnowledgeBaseConfiguration': {
                            'embeddingModelArn': embeddingModelArn,
                            'embeddingModelConfiguration': {
                                'bedrockEmbeddingModelConfiguration': {
                                    'dimensions': 1024
                                }
                            }
                        }
                    },
                    storageConfiguration={
                        'type': 'OPENSEARCH_SERVERLESS',
                        'opensearchServerlessConfiguration': {
                            'collectionArn': collectionArn,
                            'fieldMapping': {
                                'metadataField': 'AMAZON_BEDROCK_METADATA',
                                'textField': 'AMAZON_BEDROCK_TEXT',
                                'vectorField': 'vector_field'
                            },
                            'vectorIndexName': vectorIndexName
                        }
                    }                
                )   
                print('(create_knowledge_base) response: ', response)
            
                if 'knowledgeBaseId' in response['knowledgeBase']:
                    knowledge_base_id = response['knowledgeBase']['knowledgeBaseId']
                    break
                else:
                    knowledge_base_id = ""    
            except Exception:
                    err_msg = traceback.format_exc()
                    print('error message: ', err_msg)
                    time.sleep(5)
                    print(f"retrying... ({atempt})")
                    #raise Exception ("Not able to create the knowledge base")      
                
    print(f"knowledge_base_name: {knowledge_base_name}, knowledge_base_id: {knowledge_base_id}")    
    
    #########################
    # data source      
    #########################
    data_source_name = s3_bucket  
    try: 
        response = client.list_data_sources(
            knowledgeBaseId=knowledge_base_id,
            maxResults=10
        )        
        print('(list_data_sources) response: ', response)
        
        if 'dataSourceSummaries' in response:
            for data_source in response['dataSourceSummaries']:
                print('data_source: ', data_source)
                if data_source['name'] == data_source_name:
                    data_source_id = data_source['dataSourceId']
                    print('data_source_id: ', data_source_id)
                    break    
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)
        
    if not data_source_id:
        print('creating data source...')  
        try:
            response = client.create_data_source(
                dataDeletionPolicy='DELETE',
                dataSourceConfiguration={
                    's3Configuration': {
                        'bucketArn': s3_arn,
                        'inclusionPrefixes': [ 
                            s3_prefix+'/',
                        ]
                    },
                    'type': 'S3'
                },
                description = f"S3 data source: {s3_bucket}",
                knowledgeBaseId = knowledge_base_id,
                name = data_source_name,
                vectorIngestionConfiguration={
                    'chunkingConfiguration': {
                        'chunkingStrategy': 'HIERARCHICAL',
                        'hierarchicalChunkingConfiguration': {
                            'levelConfigurations': [
                                {
                                    'maxTokens': 1500
                                },
                                {
                                    'maxTokens': 300
                                }
                            ],
                            'overlapTokens': 60
                        }
                    },
                    'parsingConfiguration': {
                        'bedrockFoundationModelConfiguration': {
                            'modelArn': parsingModelArn
                        },
                        'parsingStrategy': 'BEDROCK_FOUNDATION_MODEL'
                    }
                }
            )
            print('(create_data_source) response: ', response)
            
            if 'dataSource' in response:
                if 'dataSourceId' in response['dataSource']:
                    data_source_id = response['dataSource']['dataSourceId']
                    print('data_source_id: ', data_source_id)
                    
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            #raise Exception ("Not able to create the data source")
    
    print(f"data_source_name: {data_source_name}, data_source_id: {data_source_id}")
            
initiate_knowledge_base()

def retrieve_documents_from_knowledge_base(query, top_k):
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID"   # SEMANTIC
            }},
            region_name=bedrock_region
        )
        
        try: 
            documents = retriever.invoke(query)
            # print('documents: ', documents)
            print('--> docs from knowledge base')
            for i, doc in enumerate(documents):
                print_doc(i, doc)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)    
            raise Exception ("Not able to request to LLM: "+err_msg)
        
        relevant_docs = []
        for doc in documents:
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
            print('url:', url)
            
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
    return relevant_docs

def sync_data_source():
    if knowledge_base_id and data_source_id:
        try:
            client = boto3.client(
                service_name='bedrock-agent',
                region_name=bedrock_region                
            )
            response = client.start_ingestion_job(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id
            )
            print('(start_ingestion_job) response: ', response)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)

def get_rag_prompt(text):
    # print("###### get_rag_prompt ######")
    chat = get_chat()
    # print('model_type: ', model_type)
    
    if model_type == "nova":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Provide a concise answer to the question at the end using reference texts." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
            )    
    
        human = (
            "Question: {question}"

            "Reference texts: "
            "{context}"
        ) 
        
    elif model_type == "claude":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "결과는 <result> tag를 붙여주세요."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Here is pieces of context, contained in <context> tags." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
                "Put it in <result> tags."
            )    

        human = (
            "<question>"
            "{question}"
            "</question>"

            "<context>"
            "{context}"
            "</context>"
        )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    rag_chain = prompt | chat

    return rag_chain

def run_rag_with_knowledge_base(text, st):
    global contentList
    contentList = []

    msg = ""
    top_k = numberOfDocs
    
    # retrieve
    if debug_mode == "Enable":
        st.info(f"RAG 검색을 수행합니다. 검색어: {text}")  
    
    relevant_docs = retrieve_documents_from_knowledge_base(text, top_k=top_k)
    # relevant_docs += retrieve_documents_from_tavily(text, top_k=top_k)

    # grade   
    if debug_mode == "Enable":
        st.info(f"가져온 {len(relevant_docs)}개의 문서를 평가하고 있습니다.") 

    # docs = []
    # for doc in relevant_docs:
    #     chat = get_chat()
    #     if not isKorean(doc.page_content):
    #         translated_content = traslation(chat, doc.page_content, "English", "Korean")
    #         doc.page_content = translated_content
    #         print("doc.page_content: ", doc.page_content)
    #     docs.append(doc)
    # print('translated relevant docs: ', docs)

    filtered_docs = grade_documents(text, relevant_docs)
    
    filtered_docs = check_duplication(filtered_docs) # duplication checker

    if debug_mode == "Enable":
        st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")
    
    # generate
    if debug_mode == "Enable":
        st.info(f"결과를 생성중입니다.")
    relevant_context = ""
    for document in filtered_docs:
        relevant_context = relevant_context + document.page_content + "\n\n"        
    # print('relevant_context: ', relevant_context)

    rag_chain = get_rag_prompt(text)
                       
    msg = ""    
    try: 
        result = rag_chain.invoke(
            {
                "question": text,
                "context": relevant_context                
            }
        )
        print('result: ', result)

        msg = result.content        
        if msg.find('<result>')!=-1:
            msg = msg[msg.find('<result>')+8:msg.find('</result>')]
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    reference = ""
    if filtered_docs:
        reference = get_references(filtered_docs)

    return msg+reference, filtered_docs

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
    print('timestr:', timestr)
    
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
                
    chat = get_chat()
    if isKorean(city):
        place = traslation(chat, city, "Korean", "English")
        print('city (translated): ', place)
    else:
        place = city
        city = traslation(chat, city, "English", "Korean")
        print('city (translated): ', city)
        
    print('place: ', place)
    
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
            print('result: ', result)
        
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
            print('error message: ', err_msg)   
            # raise Exception ("Not able to request to LLM")    
        
    print('weather_str: ', weather_str)                            
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
    print("###### search_by_knowledge_base ######")    
    
    global reference_docs
 
    print('keyword: ', keyword)
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    print('modified keyword: ', keyword)
    
    top_k = numberOfDocs
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID"   # SEMANTIC
            }},
        )
        
        docs = retriever.invoke(keyword)
        # print('docs: ', docs)
        print('--> docs from knowledge base')
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
    filtered_docs = grade_documents(keyword, relevant_docs)

    filtered_docs = check_duplication(filtered_docs) # duplication checker

    relevant_context = ""
    for i, document in enumerate(filtered_docs):
        print(f"{i}: {document}")
        if document.page_content:
            relevant_context += document.page_content + "\n\n"        
    print('relevant_context: ', relevant_context)
    
    if len(filtered_docs):
        reference_docs += filtered_docs
        return relevant_context
    else:        
        # relevant_context = "No relevant documents found."
        relevant_context = "관련된 정보를 찾지 못하였습니다."
        print(f"--> {relevant_context}")
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
                print('output: ', output)
                raise Exception ("Not able to request to tavily")
            else:        
                print('tavily output: ', output)
                if output == "HTTPError('429 Client Error: Too Many Requests for url: https://api.tavily.com/search')":            
                    raise Exception ("Not able to request to tavily")
                
                for result in output:
                    print('result: ', result)
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
            print('error message: ', err_msg)           
            # raise Exception ("Not able to request to tavily")  

    if answer == "":
        # answer = "No relevant documents found." 
        answer = "관련된 정보를 찾지 못하였습니다."
                     
    return answer

tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_knowledge_base]        

####################### LangGraph #######################
# Chat Agent Executor
#########################################################
def run_agent_executor(query, st):
    chatModel = get_chat()     
    model = chatModel.bind_tools(tools)

    class State(TypedDict):
        # messages: Annotated[Sequence[BaseMessage], operator.add]
        messages: Annotated[list, add_messages]

    tool_node = ToolNode(tools)

    def should_continue(state: State) -> Literal["continue", "end"]:
        print("###### should_continue ######")

        print('state: ', state)
        messages = state["messages"]    

        last_message = messages[-1]
        print('last_message: ', last_message)
        
        # print('last_message: ', last_message)
        
        # if not isinstance(last_message, ToolMessage):
        #     return "end"
        # else:                
        #     return "continue"
        if isinstance(last_message, ToolMessage) or last_message.tool_calls:
            print(f"tool_calls: ", last_message.tool_calls)

            for message in last_message.tool_calls:
                print(f"tool name: {message['name']}, args: {message['args']}")
                # update_state_message(f"calling... {message['name']}", config)

            print(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
            return "continue"
        
        #if not last_message.tool_calls:
        else:
            print("Final: ", last_message.content)
            print("--- END ---")
            return "end"
           
    def call_model(state: State, config):
        print("###### call_model ######")
        print('state: ', state["messages"])
                
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
            print('attempt: ', attempt)
            try:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )
                chain = prompt | model
                    
                response = chain.invoke(state["messages"])
                print('call_model response: ', response)

                if isinstance(response.content, list):            
                    for re in response.content:
                        if "type" in re:
                            if re['type'] == 'text':
                                print(f"--> {re['type']}: {re['text']}")

                                status = re['text']
                                print('status: ',status)
                                
                                status = status.replace('`','')
                                status = status.replace('\"','')
                                status = status.replace("\'",'')
                                
                                print('status: ',status)
                                if status.find('<thinking>') != -1:
                                    print('Remove <thinking> tag.')
                                    status = status[status.find('<thinking>')+11:status.find('</thinking>')]
                                    print('status without tag: ', status)

                                if debug_mode=="Enable":
                                    st.info(status)
                                
                            elif re['type'] == 'tool_use':                
                                print(f"--> {re['type']}: {re['name']}, {re['input']}")

                                if debug_mode=="Enable":
                                    st.info(f"{re['type']}: {re['name']}, {re['input']}")
                            else:
                                print(re)
                        else: # answer
                            print(response.content)
                break
            except Exception:
                response = AIMessage(content="답변을 찾지 못하였습니다.")

                err_msg = traceback.format_exc()
                print('error message: ', err_msg)
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
    print("msg: ", msg)

    for i, doc in enumerate(reference_docs):
        print(f"--> {i}: {doc}")
        
    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)

    msg = extract_thinking_tag(msg, st)
    
    return msg+reference, reference_docs

####################### LangGraph #######################
# Agentic Workflow: Tool Use (partial tool을 활용)
#########################################################

def run_agent_executor2(query, st):        
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        answer: str

    tool_node = ToolNode(tools)
            
    def create_agent(chat, tools):        
        tool_names = ", ".join([tool.name for tool in tools])
        print("tool_names: ", tool_names)

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
        
        return prompt | chat.bind_tools(tools)
    
    def agent_node(state, agent, name):
        print(f"###### agent_node:{name} ######")

        last_message = state["messages"][-1]
        print('last_message: ', last_message)
        if isinstance(last_message, ToolMessage) and last_message.content=="":    
            print('last_message is empty') 
            answer = get_basic_answer(state["messages"][0].content)  
            return {
                "messages": [AIMessage(content=answer)],
                "answer": answer
            }
        
        response = agent.invoke(state["messages"])
        print('response: ', response)

        if "answer" in state:
            answer = state['answer']
        else:
            answer = ""

        if isinstance(response.content, list):      
            for re in response.content:
                if "type" in re:
                    if re['type'] == 'text':
                        print(f"--> {re['type']}: {re['text']}")

                        status = re['text']
                        if status.find('<thinking>') != -1:
                            print('Remove <thinking> tag.')
                            status = status[status.find('<thinking>')+11:status.find('</thinking>')]
                            print('status without tag: ', status)

                        if debug_mode=="Enable":
                            st.info(status)

                    elif re['type'] == 'tool_use':                
                        print(f"--> {re['type']}: name: {re['name']}, input: {re['input']}")

                        if debug_mode=="Enable":
                            st.info(f"{re['type']}: name: {re['name']}, input: {re['input']}")
                    else:
                        print(re)
                else: # answer
                    answer += '\n'+response.content
                    print(response.content)
                    break

        response = AIMessage(**response.dict(exclude={"type", "name"}), name=name)     
        print('message: ', response)
        
        return {
            "messages": [response],
            "answer": answer
        }
    
    def final_answer(state):
        print(f"###### final_answer ######")        

        answer = ""        
        if "answer" in state:
            answer = state['answer']            
        else:
            answer = state["messages"][-1].content

        if answer.find('<thinking>') != -1:
            print('Remove <thinking> tag.')
            answer = answer[answer.find('</thinking>')+12:]
        print('answer: ', answer)
        
        return {
            "answer": answer
        }
    
    chat = get_chat()
    
    execution_agent = create_agent(chat, tools)
    
    execution_agent_node = functools.partial(agent_node, agent=execution_agent, name="execution_agent")
    
    def should_continue(state: State, config) -> Literal["continue", "end"]:
        print("###### should_continue ######")
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
        
        last_message = messages[-1]        
        if not last_message.tool_calls:
            print("Final: ", last_message.content)
            print("--- END ---")
            return "end"
        else:      
            print(f"tool_calls: ", last_message.tool_calls)

            for message in last_message.tool_calls:
                print(f"tool name: {message['name']}, args: {message['args']}")
                # update_state_message(f"calling... {message['name']}", config)

            print(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
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
    print('output: ', output)

    msg = output['answer']

    return msg, reference_docs

def get_basic_answer(query):
    print('#### get_basic_answer ####')
    chat = get_chat()

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
    
    chain = prompt | chat    
    output = chain.invoke({"input": query})
    print('output.content: ', output.content)

    return output.content

####################### LangGraph #######################
# Agentic Workflow: Reflection
#########################################################
def extract_reflection(draft):
    class Reflection(BaseModel):
        missing: str = Field(description="Critique of what is missing.")
        advisable: str = Field(description="Critique of what is helpful for better answer")
        superfluous: str = Field(description="Critique of what is superfluous")

    class Research(BaseModel):
        """Provide reflection and then follow up with search queries to improve the answer."""

        reflection: Reflection = Field(description="Your reflection on the initial answer.")
        search_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current answer."
        )
    
    class ReflectionKor(BaseModel):
        missing: str = Field(description="작성된 글에 있어야하는데 빠진 내용이나 단점")
        advisable: str = Field(description="더 좋은 글이 되기 위해 추가하여야 할 내용")
        superfluous: str = Field(description="글의 길이나 스타일에 대한 비평")

    class ResearchKor(BaseModel):
        """글쓰기를 개선하기 위한 검색 쿼리를 제공합니다."""

        reflection: ReflectionKor = Field(description="작성된 글에 대한 평가")
        search_queries: list[str] = Field(
            description="도출된 비평을 해결하기 위한 3개 이내의 검색어"            
        )    

    reflection = []
    search_queries = []
    for attempt in range(5):
        try:
            chat = get_chat()
            if isKorean(draft):
                structured_llm = chat.with_structured_output(Research, include_raw=True)
            else:
                structured_llm = chat.with_structured_output(Research, include_raw=True)
            
            info = structured_llm.invoke(draft)
            print(f'attempt: {attempt}, info: {info}')
                
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                print('parsed_info: ', parsed_info)
                reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                search_queries = parsed_info.search_queries
                
                print('reflection: ', parsed_info.reflection)            
                print('search_queries: ', search_queries)      

        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg) 

        return reflection, search_queries

def extract_reflection2(draft):
    system = (
        "주어진 문장을 향상시키기 위하여 아래와 같은 항목으로 개선사항을 추출합니다."
        "missing: 작성된 글에 있어야하는데 빠진 내용이나 단점"
        "advisable: 더 좋은 글이 되기 위해 추가하여야 할 내용"
        "superfluous: 글의 길이나 스타일에 대한 비평"    
        "<result> tag를 붙여주세요."
    )
    critique_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{draft}"),
        ]
    )

    reflection = ""
    for attempt in range(5):
        try:
            chat = get_chat()
            chain = critique_prompt | chat
            result = chain.invoke({
                "draft": draft
            })
            print("result: ", result)

            output = result.content

            if output.find('<result>') != -1:
                output = output[output.find('<result>')+8:output.find('</result>')]
            print('output: ', output)

            reflection = output            
            break
                
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg) 

    # search queries
    search_queries = []
    class Queries(BaseModel):
        """Provide reflection and then follow up with search queries to improve the answer."""

        search_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current answer."
        )
    class QueriesKor(BaseModel):
        """글쓰기를 개선하기 위한 검색어를 제공합니다."""

        search_queries: list[str] = Field(
            description="주어진 비평을 해결하기 위한 3개 이내의 검색어"            
        )    

    system = (
        "당신은 주어진 Draft를 개선하여 더 좋은 글쓰기를 하고자 합니다."
        "주어진 비평을 반영하여 초안을 개선하기 위한 3개 이내의 검색어를 추천합니다."
    )
    queries_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Draft: {draft} \n\n Critiques: {reflection}"),
        ]
    )
    for attempt in range(5):
        try:
            chat = get_chat()
            if isKorean(draft):
                structured_llm_queries = chat.with_structured_output(QueriesKor, include_raw=True)
            else:
                structured_llm_queries = chat.with_structured_output(Queries, include_raw=True)

            retrieval_quries = queries_prompt | structured_llm_queries
            
            info = retrieval_quries.invoke({
                "draft": draft,
                "reflection": reflection
            })
            print(f'attempt: {attempt}, info: {info}')
                
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                print('parsed_info: ', parsed_info)
                search_queries = parsed_info.search_queries
                print("search_queries: ", search_queries)
            break
                
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg) 

    return reflection, search_queries

def run_reflection(query, st):
    class State(TypedDict):
        task: str
        draft: str
        reflection: list
        search_queries: list
            
    def generate(state: State, config):    
        print("###### generate ######")
        print('task: ', state['task'])

        global reference_docs

        query = state['task']

        # grade   
        if debug_mode == "Enable":
            st.info(f"초안(draft)를 생성하기 위하여, RAG를 조회합니다.") 

        top_k = 4
        relevant_docs = retrieve_documents_from_knowledge_base(query, top_k=top_k)
    
        # grade   
        if debug_mode == "Enable":
            st.info(f"가져온 {len(relevant_docs)}개의 문서를 평가하고 있습니다.") 

        filtered_docs = grade_documents(query, relevant_docs)    
        filtered_docs = check_duplication(filtered_docs) # duplication checker
        if len(filtered_docs):
            reference_docs += filtered_docs 

        if debug_mode == "Enable":
            st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")
        
        # generate
        if debug_mode == "Enable":
            st.info(f"초안을 생성중입니다.")
        
        relevant_context = ""
        for document in filtered_docs:
            relevant_context = relevant_context + document.page_content + "\n\n"        
        # print('relevant_context: ', relevant_context)

        rag_chain = get_rag_prompt(query)
                        
        draft = ""    
        try: 
            result = rag_chain.invoke(
                {
                    "question": query,
                    "context": relevant_context                
                }
            )
            print('result: ', result)

            draft = result.content        
            if draft.find('<result>')!=-1:
                draft = draft[draft.find('<result>')+8:draft.find('</result>')]
            
            if debug_mode=="Enable":
                st.info(f"생성된 초안: {draft}")
            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            raise Exception ("Not able to request to LLM")
        
        return {"draft":draft}
    
    def reflect(state: State, config):
        print("###### reflect ######")
        print('draft: ', state["draft"])

        draft = state["draft"]
        
        if debug_mode=="Enable":
            st.info('초안을 검토하여 부족하거나 보강할 내용을 찾고, 추가 검색어를 추출합니다.')

        reflection, search_queries = extract_reflection2(draft)
        if debug_mode=="Enable":  
            st.info(f'개선할 사항: {reflection}')
            st.info(f'추가 검색어: {search_queries}')        

        return {
            "reflection": reflection,
            "search_queries": search_queries
        }
    
    def get_revise_prompt(text):
        chat = get_chat()

        if isKorean(text):
            system = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
                "draft를 critique과 information 참조하여 수정하세오."
                "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
            )
            human = (
                "draft:"
                "{draft}"
                            
                "critique:"
                "{reflection}"

                "information:"
                "{content}"
            )
        else:
            system = (
                "You are an excellent writing assistant." 
                "Revise this draft using the critique and additional information."
                "Provide the final answer with <result> tag."
            )
            human = (
                "draft:"
                "{draft}"
                            
                "critique:"
                "{reflection}"

                "information:"
                "{content}"
            )                    
        revise_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system),
                ("human", human),
            ]
        )                    
        revise_chain = revise_prompt | chat

        return revise_chain
    
    def revise_answer(state: State, config):           
        print("###### revise_answer ######")

        if debug_mode=="Enable":
            st.info("개선할 사항을 반영하여 답변을 생성중입니다.")
        
        top_k = 2        
        selected_docs = []
        for q in state["search_queries"]:
            relevant_docs = []
            filtered_docs = []
            if debug_mode=="Enable":
                st.info(f"검색을 수행합니다. 검색어: {q}")
        
            relevant_docs = retrieve_documents_from_knowledge_base(q, top_k)
            relevant_docs += retrieve_documents_from_tavily(q, top_k)

            # grade   
            if debug_mode == "Enable":
                st.info(f"가져온 {len(relevant_docs)}개의 문서를 평가하고 있습니다.") 

            filtered_docs += grade_documents(q, relevant_docs) # grading    

            if debug_mode == "Enable":
                st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")

            selected_docs += filtered_docs

        selected_docs += check_duplication(selected_docs) # check duplication
        
        global reference_docs
        if relevant_docs:
            reference_docs += selected_docs

        if debug_mode == "Enable":
            st.info(f"최종으로 {len(reference_docs)}개의 문서가 선택되었습니다.")

        content = ""
        if len(relevant_docs):
            for d in relevant_docs:
                content += d.page_content+'\n\n'
            print('content: ', content)

        for attempt in range(5):
            print(f'attempt: {attempt}')

            revise_chain = get_revise_prompt(state['task'])
            try:
                res = revise_chain.invoke(
                    {
                        "draft": state['draft'],
                        "reflection": state["reflection"],
                        "content": content
                    }
                )
                output = res.content
                print('output: ', output)

                if output.find('<result>')==-1:
                    draft = output
                else:
                    draft = output[output.find('<result>')+8:output.find('</result>')]

                print('revised_answer: ', draft)
                break

            except Exception:
                err_msg = traceback.format_exc()
                print('error message: ', err_msg)
                
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "draft": draft, 
            "revision_number": revision_number + 1
        }
    
    MAX_REVISIONS = 1
    def should_continue(state: State, config):
        print("###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"

    def buildReflection():    
        workflow = StateGraph(State)

        workflow.add_node("generate", generate)
        workflow.add_node("reflect", reflect)
        workflow.add_node("revise_answer", revise_answer)

        workflow.set_entry_point("generate")

        workflow.add_conditional_edges(
            "revise_answer", 
            should_continue, 
            {
                "end": END, 
                "continue": "reflect"}
        )

        workflow.add_edge("generate", "reflect")
        workflow.add_edge("reflect", "revise_answer")
        
        app = workflow.compile()
        
        return app
    
    # initiate
    global contentList, reference_docs
    contentList = []
    reference_docs = []

    # workflow
    app = buildReflection()
        
    inputs = {
        "task": query
    } 
    config = {
        "recursion_limit": 50,
        "max_revisions": MAX_REVISIONS,
        "parallel_processing": multi_region
    }
    
    output = app.invoke(inputs, config)
    print('output: ', output)
        
    msg = output["draft"]

    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)

    return msg+reference, reference_docs

####################### LangGraph #######################
# Agentic Workflow: Reflection (run_knowledge_guru)
#########################################################

def init_enhanced_search(st):
    chat = get_chat() 

    model = chat.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    tool_node = ToolNode(tools)

    def should_continue(state: State) -> Literal["continue", "end"]:
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
            
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:                
            return "continue"

    def call_model(state: State, config):
        print('##### call_model #####')

        messages = state["messages"]
        # print('messages: ', messages)

        last_message = messages[-1]
        print('last_message: ', last_message)

        if isinstance(last_message, ToolMessage) and last_message.content=="":              
            print('last_message is empty')      
            print('question: ', state["messages"][0].content)
            answer = get_basic_answer(state["messages"][0].content)          
            return {"messages": [AIMessage(content=answer)]}
            
        if isKorean(messages[0].content)==True:
            system = (
                "당신은 질문에 답변하기 위한 정보를 수집하는 연구원입니다."
                "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "최종 답변에는 조사한 내용을 반드시 포함하여야 하고, <result> tag를 붙여주세요."
            )
        else: 
            system = (            
                "You are a researcher charged with providing information that can be used when making answer."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will be acting as a thoughtful advisor."
                "Put it in <result> tags."
            )
                
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chain = prompt | model
                
        response = chain.invoke(messages)
        print('call_model response: ', response)
              
        # state messag
        if response.tool_calls:
            print('tool_calls response: ', response.tool_calls)

            toolinfo = response.tool_calls[-1]            
            if toolinfo['type'] == 'tool_call':
                print('tool name: ', toolinfo['name'])         

            if debug_mode=="Enable":
                st.info(f"{response.tool_calls[-1]['name']}: {response.tool_calls[-1]['args']}")
                   
        return {"messages": [response]}

    def buildChatAgent():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
            
        workflow.set_entry_point("agent")
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
    
    return buildChatAgent()

def enhanced_search(query, config, st):
    print("###### enhanced_search ######")
    inputs = [HumanMessage(content=query)]

    app_enhanced_search = init_enhanced_search(st)        
    result = app_enhanced_search.invoke({"messages": inputs}, config)   
    print('result: ', result)
            
    message = result["messages"][-1]
    print('enhanced_search: ', message)

    if message.content.find('<result>')==-1:
        return message.content
    else:
        return message.content[message.content.find('<result>')+8:message.content.find('</result>')]
    
def run_knowledge_guru(query, st):
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        reflection: list
        search_queries: list
            
    def generate(state: State, config):    
        print("###### generate ######")
        print('state: ', state["messages"])
        print('task: ', state['messages'][0].content)

        if debug_mode=="Enable":
            st.info(f"검색을 수행합니다. 검색어: {state['messages'][0].content}")
        
        draft = enhanced_search(state['messages'][0].content, config, st)  
        print('draft: ', draft)

        if debug_mode=="Enable":
            st.info(f"생성된 초안: {draft}")
        
        return {
            "messages": [AIMessage(content=draft)]
        }
    
    class Reflection(BaseModel):
        missing: str = Field(description="Critique of what is missing.")
        advisable: str = Field(description="Critique of what is helpful for better answer")
        superfluous: str = Field(description="Critique of what is superfluous")

    class Research(BaseModel):
        """Provide reflection and then follow up with search queries to improve the answer."""

        reflection: Reflection = Field(description="Your reflection on the initial answer.")
        search_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current answer."
        )
    
    class ReflectionKor(BaseModel):
        missing: str = Field(description="작성된 글에 있어야하는데 빠진 내용이나 단점")
        advisable: str = Field(description="더 좋은 글이 되기 위해 추가하여야 할 내용")
        superfluous: str = Field(description="글의 길이나 스타일에 대한 비평")

    class ResearchKor(BaseModel):
        """글쓰기를 개선하기 위한 검색 쿼리를 제공합니다."""

        reflection: ReflectionKor = Field(description="작성된 글에 대한 평가")
        search_queries: list[str] = Field(
            description="도출된 비평을 해결하기 위한 3개 이내의 검색어"            
        )    

    def reflect(state: State, config):
        print("###### reflect ######")
        print('state: ', state["messages"])    
        print('draft: ', state["messages"][-1].content)
        
        if debug_mode=="Enable":
            st.info('초안을 검토하여 부족하거나 보강할 내용을 찾고, 추가 검색어를 추출합니다.')

        reflection = []
        search_queries = []
        for attempt in range(5):
            try:
                chat = get_chat()
                if isKorean(state["messages"][-1].content):
                    structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
                else:
                    structured_llm = chat.with_structured_output(Research, include_raw=True)
                
                info = structured_llm.invoke(state["messages"][-1].content)
                print(f'attempt: {attempt}, info: {info}')
                    
                if not info['parsed'] == None:
                    parsed_info = info['parsed']
                    # print('reflection: ', parsed_info.reflection)                
                    reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                    search_queries = parsed_info.search_queries
                    
                    print('reflection: ', parsed_info.reflection)            
                    print('search_queries: ', search_queries)      

                    if debug_mode=="Enable":  
                        st.info(f'개선할 사항: {parsed_info.reflection}')
                        st.info(f'추가 검색어: {search_queries}')        
                    break
            except Exception:
                err_msg = traceback.format_exc()
                print('error message: ', err_msg) 
        
        return {
            "messages": state["messages"],
            "reflection": reflection,
            "search_queries": search_queries
        }
    
    def get_revise_prompt(text):
        chat = get_chat()

        if isKorean(text):
            system = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
                "critique과 information 참조하여 답변을 수정하십시오."
                "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
            )
            human = (
                "critique:"
                "{reflection}"

                "information:"
                "{content}"
            )
        else:
            system = (
                "You are an excellent writing assistant." 
                "Revise this draft using the critique and additional information."
                "Provide the final answer with <result> tag."
            )
            human = (
                "critique:"
                "{reflection}"

                "information:"
                "{content}"
            )                    
        revise_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system),
                MessagesPlaceholder(variable_name="messages"),
                ("human", human),
            ]
        )                    
        revise_chain = revise_prompt | chat

        return revise_chain
    
    def revise_answer(state: State, config):   
        print("###### revise_answer ######")
        
        if debug_mode=="Enable":
            st.info("개선할 사항을 반영하여 답변을 생성중입니다.")
                    
        content = []        
        if useEnhancedSearch: # search agent
            for q in state["search_queries"]:
                response = enhanced_search(q, config)
                # print(f'q: {q}, response: {response}')
                content.append(response)                   
        else:
            search = TavilySearchResults(
                max_results=2,
                include_answer=True,
                include_raw_content=True,
                api_wrapper=tavily_api_wrapper,
                search_depth="advanced", 
                # include_domains=["google.com", "naver.com"]
            )
            for q in state["search_queries"]:
                response = search.invoke(q)     
                for r in response:
                    if 'content' in r:
                        content.append(r['content'])     

        for attempt in range(5):
            print(f'attempt: {attempt}')
            messages = state["messages"]
            cls_map = {"ai": HumanMessage, "human": AIMessage}
            translated = [messages[0]] + [
                cls_map[msg.type](content=msg.content) for msg in messages[1:]
            ]
            print('translated: ', translated)     
            
            revise_chain = get_revise_prompt(content)
            try:
                res = revise_chain.invoke(
                    {
                        "messages": translated,
                        "reflection": state["reflection"],
                        "content": content
                    }
                )    
                output = res.content

                if output.find('<result>')==-1:
                    answer = output
                else:
                    answer = output[output.find('<result>')+8:output.find('</result>')]
            
                response = HumanMessage(content=answer)
                print('revised_answer: ', response.content)            
                break

            except Exception:
                err_msg = traceback.format_exc()
                print('error message: ', err_msg)            
                
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "messages": [response], 
            "revision_number": revision_number + 1
        }
    
    MAX_REVISIONS = 1
    def should_continue(state: State, config):
        print("###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"

    def buildKnowledgeGuru():    
        workflow = StateGraph(State)

        workflow.add_node("generate", generate)
        workflow.add_node("reflect", reflect)
        workflow.add_node("revise_answer", revise_answer)

        workflow.set_entry_point("generate")

        workflow.add_conditional_edges(
            "revise_answer", 
            should_continue, 
            {
                "end": END, 
                "continue": "reflect"}
        )

        workflow.add_edge("generate", "reflect")
        workflow.add_edge("reflect", "revise_answer")
        
        app = workflow.compile()
        
        return app
    
    # initiate
    global contentList, reference_docs
    contentList = []
    reference_docs = []

    # workflow
    app = buildKnowledgeGuru()
        
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50,
        "max_revisions": MAX_REVISIONS,
        "parallel_processing": parallel_processing
    }
    
    for output in app.stream({"messages": inputs}, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)
            
    print('value: ', value)
        
    msg = value["messages"][-1].content

    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)

    return msg+reference, reference_docs


####################### LangGraph #######################
# Agentic Workflow: Planning (Advanced CoT)
#########################################################
def run_planning(query, st):
    class State(TypedDict):
        input: str
        plan: list[str]
        past_steps: Annotated[List[Tuple], operator.add]
        info: Annotated[List[Tuple], operator.add]
        answer: str

    def plan_node(state: State, config):
        print("###### plan ######")
        print('input: ', state["input"])

        if debug_mode=="Enable":
            st.info(f"계획을 생성합니다. 요청사항: {state['input']}")
        
        system = (
            "당신은 user의 question을 해결하기 위해 step by step plan을 생성하는 AI agent입니다."                
            
            "문제를 충분히 이해하고, 문제 해결을 위한 계획을 다음 형식으로 4단계 이하의 계획을 세웁니다."                
            "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
            "1. [질문을 해결하기 위한 단계]"
            "2. [질문을 해결하기 위한 단계]"
            "..."                
        )
        
        human = (
            "{question}"
        )
                            
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )
        chat = get_chat()
        planner = planner_prompt | chat
        response = planner.invoke({
            "question": state["input"]
        })
        print('response.content: ', response.content)
        result = response.content
        
        #output = result[result.find('<result>')+8:result.find('</result>')]
        output = result
        
        plan = output.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')
        print('planning_steps: ', planning_steps)

        if debug_mode=="Enable":
            st.info(f"생성된 계획: {planning_steps}")
        
        return {
            "input": state["input"],
            "plan": planning_steps
        }
    
    def generate_answer(chat, relevant_docs, text):    
        relevant_context = ""
        for document in relevant_docs:
            relevant_context = relevant_context + document.page_content + "\n\n"        
        # print('relevant_context: ', relevant_context)

        if debug_mode=="Enable":
            st.info(f"계획을 수행합니다. 현재 계획 {text}")

        # generating
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "결과는 <result> tag를 붙여주세요."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Provide a concise answer to the question at the end using reference texts." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            )    
        human = (
            "Question: {input}"

            "Reference texts: "
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        # print('prompt: ', prompt)
                        
        chain = prompt | chat
        
        response = chain.invoke({
            "context": relevant_context,
            "input": text,
        })
        # print('response.content: ', response.content)

        if response.content.find('<result>') == -1:
            output = response.content
        else:
            output = response.content[response.content.find('<result>')+8:response.content.find('</result>')]        
        # print('output: ', output)
            
        return output

    def execute_node(state: State, config):
        print("###### execute ######")
        print('input: ', state["input"])
        plan = state["plan"]
        print('plan: ', plan) 
        
        chat = get_chat()

        if debug_mode=="Enable":
            st.info(f"검색을 수행합니다. 검색어 {plan[0]}")
        
        # retrieve
        relevant_docs = retrieve_documents_from_knowledge_base(plan[0], top_k=4)
        relevant_docs += retrieve_documents_from_tavily(plan[0], top_k=4)
        
        # grade   
        if debug_mode == "Enable":
            st.info(f"가져온 {len(relevant_docs)}개의 문서를 평가하고 있습니다.") 

        filtered_docs = grade_documents(plan[0], relevant_docs) # grading    
        filtered_docs = check_duplication(filtered_docs) # check duplication

        global reference_docs
        if len(filtered_docs):
            reference_docs += filtered_docs

        if debug_mode == "Enable":
            st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")
                
        # generate
        if debug_mode == "Enable":
            st.info(f"결과를 생성중입니다.")

        result = generate_answer(chat, relevant_docs, plan[0])
        
        print('task: ', plan[0])
        print('executor output: ', result)

        if debug_mode=="Enable":
            st.info(f"현 단계의 결과 {result}")
        
        # print('plan: ', state["plan"])
        # print('past_steps: ', task)        
        return {
            "input": state["input"],
            "plan": state["plan"],
            "info": [result],
            "past_steps": [plan[0]],
        }
            
    def replan_node(state: State, config):
        print('#### replan ####')
        print('state of replan node: ', state)

        if len(state["plan"]) == 1:
            print('last plan: ', state["plan"])
            return {"response":state["info"][-1]}
        
        if debug_mode=="Enable":
            st.info(f"새로운 계획을 생성합니다.")
        
        system = (
            "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
            "당신은 다음의 Question에 대한 적절한 답변을 얻고자합니다."
        )        
        human = (
            "Question: {input}"
                        
            "당신의 원래 계획은 아래와 같습니다." 
            "Original Plan:"
            "{plan}"

            "완료한 계획는 아래와 같습니다."
            "Past steps:"
            "{past_steps}"
            
            "당신은 Original Plan의 원래 계획을 상황에 맞게 수정하세요."
            "계획에 아직 해야 할 단계만 추가하세요. 이전에 완료한 계획는 계획에 포함하지 마세요."                
            "수정된 계획에는 <plan> tag를 붙여주세요."
            "만약 더 이상 계획을 세우지 않아도 Question에 답변할 수 있다면, 최종 결과로 Question에 대한 답변을 <result> tag를 붙여 전달합니다."
            
            "수정된 계획의 형식은 아래와 같습니다."
            "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
            "1. [질문을 해결하기 위한 단계]"
            "2. [질문을 해결하기 위한 단계]"
            "..."         
        )                   
        
        replanner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )     
        
        chat = get_chat()
        replanner = replanner_prompt | chat
        
        response = replanner.invoke({
            "input": state["input"],
            "plan": state["plan"],
            "past_steps": state["past_steps"]
        })
        print('replanner output: ', response.content)
        result = response.content

        if result.find('<plan>') == -1:
            return {"response":response.content}
        else:
            output = result[result.find('<plan>')+6:result.find('</plan>')]
            print('plan output: ', output)

            plans = output.strip().replace('\n\n', '\n')
            planning_steps = plans.split('\n')
            print('planning_steps: ', planning_steps)

            if debug_mode=="Enable":
                st.info(f"새로운 계획: {planning_steps}")

            return {"plan": planning_steps}
        
    def should_end(state: State) -> Literal["continue", "end"]:
        print('#### should_end ####')
        # print('state: ', state)
        
        if "response" in state and state["response"]:
            print('response: ', state["response"])            
            next = "end"
        else:
            print('plan: ', state["plan"])
            next = "continue"
        print(f"should_end response: {next}")
        
        return next
        
    def final_answer(state: State) -> str:
        print('#### final_answer ####')
        
        # get final answer
        context = "".join(f"{info}\n" for info in state['info'])
        print('context: ', context)
        
        query = state['input']
        print('query: ', query)

        if debug_mode=="Enable":
            st.info(f"최종 답변을 생성합니다.")
        
        if isKorean(query)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                #"결과는 <result> tag를 붙여주세요."
            )
        else: 
            system = (
                "Here is pieces of context, contained in <context> tags."
                "Provide a concise answer to the question at the end."
                "Explains clearly the reason for the answer."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                #"Put it in <result> tags."
            )
    
        human = (
            "Reference texts:"
            "{context}"

            "Question: {input}"
        )
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        # print('prompt: ', prompt)
                    
        chat = get_chat()
        chain = prompt | chat
        
        try: 
            response = chain.invoke(
                {
                    "context": context,
                    "input": query,
                }
            )
            result = response.content

            if result.find('<result>')==-1:
                output = result
            else:
                output = result[result.find('<result>')+8:result.find('</result>')]
                
            print('output: ', output)
            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)      
            
        return {"answer": output}  

    def buildPlanAndExecute():
        workflow = StateGraph(State)
        workflow.add_node("planner", plan_node)
        workflow.add_node("executor", execute_node)
        workflow.add_node("replaner", replan_node)
        workflow.add_node("final_answer", final_answer)
        
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "replaner")
        workflow.add_conditional_edges(
            "replaner",
            should_end,
            {
                "continue": "executor",
                "end": "final_answer",
            },
        )
        workflow.add_edge("final_answer", END)

        return workflow.compile()

    # initiate
    global contentList, reference_docs
    contentList = []
    reference_docs = []

    # workflow
    app = buildPlanAndExecute()    
        
    inputs = {"input": query}
    config = {
        "recursion_limit": 50
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)            
    print('value: ', value)

    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)
    
    return value["answer"]+reference, reference_docs

    
####################### LangGraph #######################
# Agentic Workflow Multi-agent Collaboration 
#########################################################
def run_long_form_writing_agent(query, st):
    # Workflow - Reflection
    class ReflectionState(TypedDict):
        draft : str
        reflection : List[str]
        search_queries : List[str]
        revised_draft: str
        revision_number: int
        reference: List[str]
                
    def reflect_node(state: ReflectionState, config):
        class ReflectionKor(BaseModel):
            missing: str = Field(description="작성된 글에 있어야하는데 빠진 내용이나 단점")
            advisable: str = Field(description="더 좋은 글이 되기 위해 추가하여야 할 내용")
            superfluous: str = Field(description="글의 길이나 스타일에 대한 비평")

        class ResearchKor(BaseModel):
            """글쓰기를 개선하기 위한 검색 쿼리를 제공합니다."""

            reflection: ReflectionKor = Field(description="작성된 글에 대한 평가")
            search_queries: list[str] = Field(
                description="도출된 비평을 해결하기 위한 3개 이내의 검색어"            
            )    

        class Reflection(BaseModel):
            missing: str = Field(description="Critique of what is missing.")
            advisable: str = Field(description="Critique of what is helpful for better writing")
            superfluous: str = Field(description="Critique of what is superfluous")

        class Research(BaseModel):
            """Provide reflection and then follow up with search queries to improve the writing."""

            reflection: Reflection = Field(description="Your reflection on the initial writing.")
            search_queries: list[str] = Field(
                description="1-3 search queries for researching improvements to address the critique of your current writing."
            )
        
        print("###### reflect ######")
        draft = state['draft']
        
        idx = config.get("configurable", {}).get("idx")
        print('reflect_node idx: ', idx)

        if debug_mode=="Enable":
            st.info(f"{idx}: draft에서 개선 사항을 도출합니다.")
    
        reflection = []
        search_queries = []
        for attempt in range(20):
            chat = get_chat()
            if isKorean(draft):
                structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
            else:
                structured_llm = chat.with_structured_output(Research, include_raw=True)
            
            try:
                # print('draft: ', draft)
                info = structured_llm.invoke(draft)
                print(f'attempt: {attempt}, info: {info}')
                    
                if not info['parsed'] == None:
                    parsed_info = info['parsed']
                    # print('reflection: ', parsed_info.reflection)                
                    reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                    search_queries = parsed_info.search_queries

                    if debug_mode=="Enable":
                        st.info(f"{idx}: 개선사항: {reflection}")
                    
                    print('reflection: ', parsed_info.reflection)
                    print('search_queries: ', search_queries)
            
                    if isKorean(draft):
                        translated_search = []
                        for q in search_queries:
                            chat = get_chat()
                            if isKorean(q):
                                search = traslation(chat, q, "Korean", "English")
                            else:
                                search = traslation(chat, q, "English", "Korean")
                            translated_search.append(search)
                            
                        print('translated_search: ', translated_search)
                        search_queries += translated_search

                    if debug_mode=="Enable":
                        st.info(f"검색어: {search_queries}")

                    print('search_queries (mixed): ', search_queries)
                    break
            except Exception:
                print('---> parsing error')

                err_msg = traceback.format_exc()
                print('error message: ', err_msg)
                # raise Exception ("Not able to request to LLM")
            
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "reflection": reflection,
            "search_queries": search_queries,
            "revision_number": revision_number + 1
        }
    
    def reflect_node2(state: ReflectionState, config):        
        print("###### reflect ######")
        draft = state['draft']
        
        idx = config.get("configurable", {}).get("idx")
        print('reflect_node idx: ', idx)

        if debug_mode=="Enable":
            st.info(f"{idx}: draft에서 개선 사항을 도출합니다.")
    
        reflection, search_queries = extract_reflection2(draft)
        if debug_mode=="Enable":  
            st.info(f'개선할 사항: {reflection}')
            st.info(f'추가 검색어: {search_queries}')    

        if isKorean(draft):
            translated_search = []
            for q in search_queries:
                chat = get_chat()
                if isKorean(q):
                    search = traslation(chat, q, "Korean", "English")
                else:
                    search = traslation(chat, q, "English", "Korean")
                translated_search.append(search)
                
            print('translated_search: ', translated_search)
            search_queries += translated_search

        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "reflection": reflection,
            "search_queries": search_queries,
            "revision_number": revision_number + 1
        }

    def retrieve_for_writing(conn, q, config):
        idx = config.get("configurable", {}).get("idx") 
         
        if debug_mode=="Enable":
            st.info(f"검색을 수행합니다. 검색어: {q}")

        relevant_docs = retrieve_documents_from_knowledge_base(q, top_k=numberOfDocs)
        relevant_docs += retrieve_documents_from_tavily(q, top_k=numberOfDocs)

        # translate
        # docs = []
        # for doc in relevant_docs:
        #     chat = get_chat()
        #     if not isKorean(doc.page_content):
        #         translated_content = traslation(chat, doc.page_content, "English", "Korean")
        #         doc.page_content = translated_content
        #     docs.append(doc)
        # print('translated relevant docs: ', docs)
                
        # grade
        filtered_docs = grade_documents(q, relevant_docs) # grading    
        filtered_docs = check_duplication(filtered_docs) # check duplication
                                
        conn.send(filtered_docs)
        conn.close()

    def parallel_retriever(search_queries, config):
        relevant_documents = []    
        
        processes = []
        parent_connections = []
        for q in search_queries:
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
                
            process = Process(target=retrieve_for_writing, args=(child_conn, q, config))
            processes.append(process)

        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            rel_docs = parent_conn.recv()

            if(len(rel_docs)>=1):
                for doc in rel_docs:
                    relevant_documents.append(doc)    

        for process in processes:
            process.join()
        
        #print('relevant_docs: ', relevant_docs)
        return relevant_documents

    def retrieve_docs(search_queries, config):
        docs = []
        
        parallel_retrieval = config.get("configurable", {}).get("parallel_retrieval")
        print('parallel_retrieval: ', parallel_retrieval)
        
        if parallel_retrieval == 'enable':
            docs = parallel_retriever(search_queries, config)
        else:
            for q in search_queries:      
                if debug_mode=="Enable":
                    st.info(f"검색을 수행합니다. 검색어: {q}")

                relevant_docs = retrieve_documents_from_knowledge_base(q, top_k=numberOfDocs)
                relevant_docs += retrieve_documents_from_tavily(q, top_k=numberOfDocs)
            
                # grade
                docs += grade_documents(q, relevant_docs) # grading
                    
            docs = check_duplication(docs) # check duplication
            for i, doc in enumerate(docs):
                print(f"#### {i}: {doc.page_content[:100]}")
                            
        return docs
        
    def get_revise_prompt(draft):
        chat = get_chat()

        if isKorean(draft):
            system = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
                "draft을 critique과 information 사용하여 수정하십시오."
                "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
            )
            human = (
                "draft:"
                "{draft}"
                            
                "critique:"
                "{reflection}"

                "information:"
                "{content}"
            )
        else:    
            system = (
                "You are an excellent writing assistant." 
                "Revise this draft using the critique and additional information."
                "Provide the final answer with <result> tag."
            )
            human = (                            
                "draft:"
                "{draft}"
                            
                "critique:"
                "{reflection}"

                "information:"
                "{content}"
            )
                    
        revise_prompt = ChatPromptTemplate([
            ('system', system),
            ('human', human)
        ])

        chat = get_chat()
        reflect_chain = revise_prompt | chat

        return reflect_chain
        
    def revise_draft(state: ReflectionState, config):   
        print("###### revise_draft ######")
        
        draft = state['draft']
        search_queries = state['search_queries']
        reflection = state['reflection']
        print('draft: ', draft)
        print('search_queries: ', search_queries)
        print('reflection: ', reflection)

        idx = config.get("configurable", {}).get("idx")
        print('revise_draft idx: ', idx)
        
        if debug_mode=="Enable":
            st.info(f"{idx}: 개선사항을 반영하여 새로운 답변을 생성합니다.")
        
        # reference = state['reference'] if 'reference' in state else []     
        if 'reference' in state:
            reference = state['reference'] 
        else:
            reference = []            

        if len(search_queries) and len(reflection):
            docs = retrieve_docs(search_queries, config)        
            print('docs: ', docs)
                    
            content = []   
            if len(docs):                
                for d in docs:
                    content.append(d.page_content)            
                print('content: ', content)
                                    
                revise_chain = get_revise_prompt(content)
                res = revise_chain.invoke(
                    {
                        "draft": draft,
                        "reflection": reflection,
                        "content": content
                    }
                )
                output = res.content
                # print('output: ', output)
                
                if output.find('<result>') == -1:
                    revised_draft = output
                else:
                    revised_draft = output[output.find('<result>')+8:output.find('</result>')]
                    
                # print('--> draft: ', draft)
                print('--> reflection: ', reflection)
                print('--> revised_draft: ', revised_draft)

                st.info(f"revised_draft: {revised_draft}")

                reference += docs
                print('len(reference): ', len(reference))
            else:
                print('No relevant document!')
                revised_draft = draft
        else:
            print('No reflection!')
            revised_draft = draft
            
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        
        return {
            "revised_draft": revised_draft,            
            "revision_number": revision_number,
            "reference": reference
        }
        
    MAX_REVISIONS = 1
    def should_continue(state: ReflectionState, config):
        print("###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"        
    
    def buildReflection():
        workflow = StateGraph(ReflectionState)

        # Add nodes
        workflow.add_node("reflect_node", reflect_node2)
        workflow.add_node("revise_draft", revise_draft)

        # Set entry point
        workflow.set_entry_point("reflect_node")
        
        workflow.add_conditional_edges(
            "revise_draft", 
            should_continue, 
            {
                "end": END, 
                "continue": "reflect_node"
            }
        )

        # Add edges
        workflow.add_edge("reflect_node", "revise_draft")
        
        return workflow.compile()
    
    # Workflow - Long Writing
    class State(TypedDict):
        instruction : str
        planning_steps : List[str]
        drafts : List[str]
        # num_steps : int
        final_doc : str
        word_count : int
            
    def plan_node(state: State, config):
        print("###### plan ######")
        instruction = state["instruction"]
        print('subject: ', instruction)

        if debug_mode=="Enable":
            st.info(f"계획을 생성합니다. 요청사항: {instruction}")
        
        if isKorean(instruction):
            system = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."
                "당신은 글쓰기 지시 사항을 여러 개의 하위 작업으로 나눌 것입니다."
                "글쓰기 계획은 5단계 이하로 작성합니다."
                "각 하위 작업은 에세이의 한 단락 작성을 안내할 것이며, 해당 단락의 주요 내용과 단어 수 요구 사항을 포함해야 합니다."
                "각 하위 작업이 명확하고 구체적인지, 그리고 모든 하위 작업이 작문 지시 사항의 전체 내용을 다루고 있는지 확인하세요."
                "과제를 너무 세분화하지 마세요. 각 하위 과제의 문단은 500단어 이상 3000단어 이하여야 합니다."
                "다른 내용은 출력하지 마십시오. 이것은 진행 중인 작업이므로 열린 결론이나 다른 수사학적 표현을 생략하십시오."     
            )
            human = (
                "글쓰기 지시 사항은 아래와 같습니다."
                "Instruction:"
                "{instruction}"
                
                "다음 형식으로 나누어 주시기 바랍니다. 각 하위 작업은 한 줄을 차지합니다:"
                "1. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [Word count requirement, e.g., 800 words]"
                "2. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [word count requirement, e.g. 1500 words]."
                "..."                                       
            )
        else:
            system = (
                "You are a helpful assistant highly skilled in long-form writing."
                "You will break down the writing instruction into multiple subtasks."
                "Writing plans are created in five steps or less."
                "Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph."
                "Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction."
                "Do not split the subtasks too finely; each subtask's paragraph should be no less than 500 words and no more than 3000 words."
                "Do not output any other content. As this is an ongoing work, omit open-ended conclusions or other rhetorical hooks."
            )
            human = (                
                "The writing instruction is as follows:"
                "<instruction>"
                "{instruction}"
                "<instruction>"
                
                "Please break it down in the following format, with each subtask taking up one line:"
                "1. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [Word count requirement, e.g., 800 words]"
                "2. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [word count requirement, e.g. 1500 words]."
                "..."                
            )
        
        planner_prompt = ChatPromptTemplate([
            ('system', system),
            ('human', human) 
        ])
                
        chat = get_chat()
        
        planner = planner_prompt | chat
    
        response = planner.invoke({"instruction": instruction})
        print('response: ', response.content)
    
        plan = response.content.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')        
        print('planning_steps: ', planning_steps)

        if debug_mode=="Enable":
            st.info(f"생성된 계획: {planning_steps}")
            
        return {
            "instruction": instruction,
            "planning_steps": planning_steps
        }
        
    def execute_node(state: State, config):
        print("###### execute_node ######")        
        
        instruction = state["instruction"]        
        if isKorean(instruction):
            system = (
                "당신은 훌륭한 글쓰기 도우미입니다." 
                "당신은 글쓰기 plan에 따라 instruction에 대한 글을 작성하고자 합니다."
                "이전 단계에서 written text까지 작성하였고, next step을 계속 작성합니다."
                "글이 끊어지지 않고 잘 이해되도록 하나의 문단을 충분히 길게 작성합니다."
                
                "필요하다면 앞에 작은 부제를 추가할 수 있습니다."
                "이미 작성된 텍스트를 반복하지 말고 작성한 문단만 출력하세요."                
                "Markdown 포맷으로 서식을 작성하세요."                
            )
            human = (
                "아래는 이전 단계에서 작성된 텍스트입니다."
                "Written text:"
                "{text}"

                "글쓰기 지시사항은 아래와 같습니다."                
                "Instruction:"
                "{intruction}"

                "전체 글쓰기 단계는 아래와 같습니다."
                "Plan:"
                "{plan}"
               
                "다음으로 작성할 글쓰기 단계입니다. Instruction, plan, written text을 참조하여 next step을 계속 작성합니다."
                "Next step:"
                "{step}"
            )
        else:    
            system = (
                "You are an excellent writing assistant." 
                "You intend to write an article about instructions according to a writing plan. "
                "In the previous step, we wrote up to the written text, and we will continue writing the next step. "
                "Please help me continue writing the next paragraph based on the writing instruction, writing steps, and the already written text."

                "If needed, you can add a small subtitle at the beginning."
                "Remember to only output the paragraph you write, without repeating the already written text."
                "Use markdown syntax to format your output:"
                "- Headings: # for main, ## for sections, ### for subsections, etc."
                "- Lists: * or - for bulleted, 1. 2. 3. for numbered"
                "- Do not repeat yourself"
                "Provide the final answer with <result> tag."
            )
            human = (
                "The text written in the previous step is as follows."
                "Written text:"
                "{text}"  

                "The writing instructions are as follows."
                "Instruction:"
                "{intruction}"

                "The entire writing plan is as follows."
                "Plan:"
                "{plan}"

                "The next step to write is as follows. Please refer to the writing instruction, writing steps, and the already written text."
                "Next step:"
                "{step}"                                          
            )

        write_prompt = ChatPromptTemplate([
            ('system', system),
            ('human', human)
        ])
        
        planning_steps = state["planning_steps"]        
        if len(planning_steps) > 50:
            print("plan is too long")
            # print(plan)
            return
        
        text = ""
        drafts = []
        for idx, step in enumerate(planning_steps):            
            # Invoke the write_chain
            chat = get_chat()
            write_chain = write_prompt | chat     

            plan = ""
            for p in planning_steps:
                plan += p + '\n'

            result = write_chain.invoke({
                "intruction": instruction,
                "plan": plan,
                "text": text,
                "step": step
            })
            output = result.content
            # print('output: ', output)

            if debug_mode=="Enable":
                st.info(f"수행단계: {step}")
            
            if output.find('<result>')==-1:
                draft = output
            else:
                draft = output[output.find('<result>')+8:output.find('</result>')]

            if debug_mode=="Enable":
                st.info(f"생성결과: {draft}")
                                              
            print(f"--> step: {step}")
            print(f"--> draft: {draft}")
                
            drafts.append(draft)
            text += draft + '\n\n'

        return {
            "instruction": instruction,
            "drafts": drafts
        }

    def reflect_draft(conn, reflection_app, idx, config, draft):     
        inputs = {
            "draft": draft
        }     

        try:        
            output = reflection_app.invoke(inputs, config)
            print('idx: ', idx)

        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            # raise Exception ("Not able to request to LLM")    
    
        result = {
            "revised_draft": output['revised_draft'],
            "idx": idx,
            "reference": output['reference']
        }
            
        conn.send(result)    
        conn.close()
        
    def reflect_drafts_using_parallel_processing(drafts, config):
        revised_drafts = drafts
        
        processes = []
        parent_connections = []
        references = []
        
        reflection_app = buildReflection()
        
        for idx, draft in enumerate(drafts):
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
            
            print(f"idx:{idx} --> draft:{draft}")
            
            app_config = {
                "recursion_limit": 50,
                "max_revisions": MAX_REVISIONS,
                "idx": idx,
                "parallel_retrieval": "enable"
            }
            process = Process(target=reflect_draft, args=(child_conn, reflection_app, idx, app_config, draft))
            processes.append(process)
            
        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            result = parent_conn.recv()

            if result is not None:
                print('result: ', result)
                revised_drafts[result['idx']] = result['revised_draft']

                if result['reference']:
                    references += result['reference']

        for process in processes:
            process.join()
                
        final_doc = ""   
        for revised_draft in revised_drafts:
            final_doc += revised_draft + '\n\n'
        
        return final_doc, references

    def get_subject(query):
        system = (
            "Extract the subject of the question in 6 words or fewer."
        )
        
        human = "<question>{question}</question>"
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        # print('prompt: ', prompt)
        
        chat = get_chat()
        chain = prompt | chat    
        try: 
            result = chain.invoke(
                {
                    "question": query
                }
            )        
            subject = result.content
            # print('the subject of query: ', subject)
            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            raise Exception ("Not able to request to LLM")        
        return subject
    
    def markdown_to_html(body):
        html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        <md-block>
        </md-block>
        <script type="module" src="https://md-block.verou.me/md-block.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.css" integrity="sha512-n5zPz6LZB0QV1eraRj4OOxRbsV7a12eAGfFcrJ4bBFxxAwwYDp542z5M0w24tKPEhKk2QzjjIpR5hpOjJtGGoA==" crossorigin="anonymous" referrerpolicy="no-referrer"/>
    </head>
    <body>
        <div class="markdown-body">
            <md-block>{body}
            </md-block>
        </div>
    </body>
    </html>"""        
        return html

    def get_references_for_markdown(docs):
        reference = ""
        nameList = []
        cnt = 1
        for i, doc in enumerate(docs):
            print(f"reference {i}: doc")
            page = ""
            if "page" in doc.metadata:
                page = doc.metadata['page']
                #print('page: ', page)            
            url = ""
            if "url" in doc.metadata:
                url = doc.metadata['url']
                #print('url: ', url)                
            name = ""
            if "name" in doc.metadata:
                name = doc.metadata['name']
                #print('name: ', name)     
            pos = name.rfind('/')
            name = name[pos+1:]
            print(f"name: {name}")
            
            excerpt = ""+doc.page_content

            excerpt = re.sub('"', '', excerpt)
            print('length: ', len(excerpt))
            
            if name in nameList:
                print('duplicated!')
            else:
                reference = reference + f"{cnt}. [{name}]({url})"
                nameList.append(name)
                cnt = cnt+1
                
        return reference
    
    def revise_answer(state: State, config):
        print("###### revise ######")
        drafts = state["drafts"]        
        print('drafts: ', drafts)
        
        parallel_revise = config.get("configurable", {}).get("parallel_revise", "enable")
        print('parallel_revise: ', parallel_revise)

        if debug_mode=="Enable":
            st.info("문서를 개선합니다.")
        
        # reflection
        if parallel_revise == 'enable':  # parallel processing
            final_doc, references = reflect_drafts_using_parallel_processing(drafts, config)
        else:
            reflection_app = buildReflection()
                
            final_doc = ""   
            references = []
                        
            for idx, draft in enumerate(drafts):
                inputs = {
                    "draft": draft
                }                    
                app_config = {
                    "recursion_limit": 50,
                    "max_revisions": MAX_REVISIONS,
                    "idx": idx,
                    "parallel_retrieval": "disable"
                }
                output = reflection_app.invoke(inputs, config=app_config)
                final_doc += output['revised_draft'] + '\n\n'
                references += output['reference']

        subject = get_subject(state['instruction'])
        subject = subject.replace(" ","_")
        subject = subject.replace("?","")
        subject = subject.replace("!","")
        subject = subject.replace(".","")
        subject = subject.replace(":","")
        
        print('len(references): ', len(references))
        
        # markdown file
        markdown_key = 'markdown/'+f"{subject}.md"
        # print('markdown_key: ', markdown_key)
        
        final_doc = f"## {state['instruction']}\n\n"+final_doc

        if references:
            print('references: ', references)

            markdown_reference = get_references(references)
            
            print('markdown_reference: ', markdown_reference)

            final_doc += markdown_reference
                
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )  
        response = s3_client.put_object(
            Bucket=s3_bucket,
            Key=markdown_key,
            ContentType='text/markdown',
            Body=final_doc.encode('utf-8')
        )
        # print('response: ', response)
        
        markdown_url = f"{path}/{markdown_key}"
        print('markdown_url: ', markdown_url)
        
        # html file
        html_key = 'markdown/'+f"{subject}.html"
            
        html_body = markdown_to_html(final_doc)
        print('html_body: ', html_body)
        
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )  
        response = s3_client.put_object(
            Bucket=s3_bucket,
            Key=html_key,
            ContentType='text/html',
            Body=html_body
        )
        # print('response: ', response)
        
        html_url = f"{path}/{html_key}"
        print('html_url: ', html_url)

        final_doc += f"\n[미리보기 링크]({html_url})\n\n[다운로드 링크 - {subject}.md]({markdown_url})"
        
        return {
            "final_doc": final_doc
        }
        
    def buildLongformWriting():
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("plan_node", plan_node)
        workflow.add_node("execute_node", execute_node)
        workflow.add_node("revise_answer", revise_answer)

        # Set entry point
        workflow.set_entry_point("plan_node")

        # Add edges
        workflow.add_edge("plan_node", "execute_node")
        workflow.add_edge("execute_node", "revise_answer")
        workflow.add_edge("revise_answer", END)
        
        return workflow.compile()
    
    # initiate
    global contentList, reference_docs
    contentList = []
    reference_docs = []
    
    # Run the workflow
    app = buildLongformWriting()    
    inputs = {
        "instruction": query
    }    
    config = {
        "recursion_limit": 100,
        "parallel_revise": multi_region
    }
    
    output = app.invoke(inputs, config)
    print('output: ', output)
    
    return output['final_doc'], reference_docs


####################### LangGraph #######################
# Agentic Solver for Korean CSAT
#########################################################
def solve_CSAT_problem(contents, st):
    json_data = json.loads(contents)
    # print('json_data: ', json_data)
        
    msg = ""  
    total_score = 0
    scores = []
    for question_group in json_data:
        problems = question_group["problems"]
        local_score = 0
        for problem in problems:
            local_score += int(problem["score"])
        total_score += local_score
        scores.append(local_score)
    print('total_score: ', total_score)    
    st.info(f'주어진 문제는 모두 {len(json_data)}개의 절을 가지고 있고, 각 절의 점수 분포는 {scores}이며, 전체 점수는 {total_score}점 입니다.')
                            
    if multi_region=="Enable":
        msg, earn_score = solve_problems_using_parallel_processing(json_data, st)

        print('score: ', earn_score)
        msg += f"\n점수: {earn_score}점 / {total_score}점\n"

    else:
        total_idx = len(json_data)+1
        earn_score = total_available_score = 0
        #for idx, question_group in enumerate(json_data[:2]):
        for idx, question_group in enumerate(json_data):
            paragraph = question_group["paragraph"]
            print('paragraph: ', paragraph)
            
            problems = question_group["problems"]
            print('problems: ', json.dumps(problems))
            
            result = solve_problems_in_paragraph(paragraph, problems, idx, total_idx, st)
            print('result: ', result)

            idx = result["idx"]
            message = result["message"]
            score = result["score"]
            available_score = result["available_score"]
            print('idx: ', idx)
            print('message: ', message)
            print('score: ', score)
            print('available_score: ', available_score)
            
            msg += message
            earn_score += score
            total_available_score += available_score
            
            msg += "\n\n"
        
            st.warning(f"{idx+1}절까지 수행한 결과는 {earn_score} / {total_available_score}점입니다.")
        
        print('score: ', earn_score)
        msg += f"\n점수: {earn_score}점 / {total_available_score}점\n"
    
    st.info(f"{msg}")

    return msg
    
def solve_problems_in_paragraph(paragraph, problems, idx, total_idx, st):
    message = f"{idx+1}/{total_idx}\n"
    
    earn_score = 0
    available_score = 0
    for n, problem in enumerate(problems):
        print(f'--> problem[{n}]: {problem}')
    
        question = problem["question"]
        print('question: ', question)
        question_plus = ""
        if "question_plus" in problem:
            question_plus = problem["question_plus"]
            print('question_plus: ', question_plus)
        choices = problem["choices"]
        print('choices: ', choices)
        correct_answer = problem["answer"]
        print('correct_answer: ', correct_answer)
        score = problem["score"]
        print('score: ', score)
        available_score += score

        selected_answer = solve_CSAT_Korean(paragraph, question, question_plus, choices, idx, n, correct_answer, score, st)
        print('selected_answer: ', selected_answer)
                
        print(f'correct_answer: {correct_answer}, selected_answer: {selected_answer}')
        if correct_answer == selected_answer:
            message += f"{question} {selected_answer} (OK)\n"
            earn_score += int(score)
        else:
            message += f"{question} {selected_answer} (NOK, {correct_answer}, -{score})\n"
                    
    print('earn_score: ', earn_score)
    print('message: ', message)

    st.warning(f"{idx+1}절의 {len(problems)}개의 문제에서 얻어진 점수는 {earn_score} / {available_score}점 입니다.")
    
    return {
        "idx": idx, 
        "message": message, 
        "score": earn_score,
        "available_score": available_score
    }

def solve_problems_using_parallel_processing(json_data, st):
    processes = []
    parent_connections = []
    
    total_idx = len(json_data)
    print('total_idx: ', total_idx)
    
    messages = []
    earn_score = 0
    for idx in range(total_idx):
        messages.append("")
        
    for idx, question_group in enumerate(json_data[:1]):
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
        
        print(f"idx:{idx} --> data:{question_group}")
        
        paragraph = question_group["paragraph"]
        # print('paragraph: ', paragraph)
                    
        problems = question_group["problems"]
        # print('problems: ', json.dumps(problems))
        
        process = Process(target=solve_problems, args=(child_conn, paragraph, problems, idx, total_idx, st))
        processes.append(process)
        
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        result = parent_conn.recv()
        print('result: ', result)
        
        idx = result["idx"]
        message = result["message"]
        score = result["score"]
        print(f"idx:{idx} --> socre: {score}, message:{message}")

        if message is not None:
            print('message: ', message)
            messages[idx] = message
            earn_score += score

    for process in processes:
        process.join()
            
    final_msg = ""   
    for message in messages:
        final_msg += message + '\n'
    
    print('earn_score: ', earn_score)
    print('final_msg: ', final_msg)
    
    return final_msg, earn_score

def solve_problems(conn, paragraph, problems, idx, total_idx, st):
    message = f"{idx+1}/{total_idx}\n"
    
    earn_score = available_score = 0    
    for n, problem in enumerate(problems):
        print(f'--> problem[{n}]: {problem}')
    
        question = problem["question"]
        print('question: ', question)
        question_plus = ""
        if "question_plus" in problem:
            question_plus = problem["question_plus"]
            print('question_plus: ', question_plus)
        choices = problem["choices"]
        print('choices: ', choices)
        correct_answer = problem["answer"]
        print('correct_answer: ', correct_answer)
        score = problem["score"]
        print('score: ', score)
        available_score += score

        selected_answer = solve_CSAT_Korean(paragraph, question, question_plus, choices, idx, n, correct_answer, score, st)
        print('selected_answer: ', selected_answer)

        print(f'correct_answer: {correct_answer}, selected_answer: {selected_answer}')
        if correct_answer == selected_answer:
            message += f"{question} {selected_answer} (OK)\n"
            earn_score += int(score)
        else:
            message += f"{question} {selected_answer} (NOK, {correct_answer}, -{score})\n"
        
        
        # if output.isnumeric():
        #     selected_answer = int(output)
        #     print('selected_answer: ', selected_answer)
        # else:
        #     class Selection(BaseModel):
        #         select: int = Field(description="선택지의 번호")
            
        #     chat = get_chat()
        #     structured_llm = chat.with_structured_output(Selection, include_raw=True)
            
        #     info = structured_llm.invoke(output)
        #     selected_answer = 0
        #     for attempt in range(5):
        #         #print(f'attempt: {attempt}, info: {info}')
        #         if not info['parsed'] == None:
        #             parsed_info = info['parsed']
        #             #print('parsed_info: ', parsed_info)
        #             selected_answer = parsed_info.select                    
        #             print('selected_answer: ', selected_answer)
        #             break
            
    print('earn_score: ', earn_score) 
    print('message: ', message)

    st.warning(f"{idx+1}절의 {len(problems)}개의 문제에서 얻어진 점수는 {earn_score} / {available_score}점 입니다.")
    
    st.info(f"{str(idx)}: {message}")
    
    conn.send({
        "idx": idx, 
        "message": message, 
        "score": earn_score,
        "available_score": available_score
    })
    
    conn.close()

def string_to_int(output):
    com = re.compile('\d') 
    value = com.findall(output)
    
    result = ""
    if not len(value) == 0:
        for v in value:
            result += v
        print('result: ', result)
        answer = int(result)
    else:
        answer = 0  # no selection

    return answer

def solve_CSAT_Korean(paragraph, question, question_plus, choices, idx, nth, correct_answer, score, st):    
    class State(TypedDict):
        plan: list[str]
        past_steps: Annotated[List[Tuple], operator.add]
        info: Annotated[List[Tuple], operator.add]
        paragraph: str
        question: str
        question_plus: str
        choices: list[str]
        answer: int

    def plan_node(state: State, config):
        print("###### plan ######")
        print('paragraph: ', state["paragraph"])
        print('question: ', state["question"])
        print('question_plus: ', state["question_plus"])

        idx = config.get("configurable", {}).get("idx")
        nth = config.get("configurable", {}).get("nth")
        score = config.get("configurable", {}).get("score")    
        paragraph = state["paragraph"]
        question_plus = state["question_plus"]
        choices = state["choices"]
        
        if question_plus:
            st.success(f"Question: ({idx}-{nth})\n\n{paragraph}\n\n{question_plus}\n\n{question} (score: {score})\n\n{choices}")
        else:
            st.success(f"Question: ({idx}-{nth})\n\n{paragraph}\n\n{question} (score: {score})\n\n{choices}")
                
        list_choices = ""
        choices = state["choices"]
        for i, choice in enumerate(choices):
            list_choices += f"({i+1}) {choice}\n"
        print('list_choices: ', list_choices)    
        
        idx = config.get("configurable", {}).get("idx")
        nth = config.get("configurable", {}).get("nth")

        notification = f"({idx}-{nth}) 계획을 생성중입니다..."
        print('notification: ', notification)
        st.info(notification)
        
        system = (
            "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
            
            "문제를 충분히 이해하고, 문제 해결을 위한 계획을 다음 형식으로 4단계 이하의 계획을 세웁니다."                
            "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
            "1. [질문을 해결하기 위한 단계]"
            "2. [질문을 해결하기 위한 단계]"
            "..."                
        )
        
        if model_type=="claude":
            human = (
                "<paragraph> tag의 주어진 문장을 참조하여 <question> tag의 주어진 질문에 대한 적절한 답변을 <choice> tag의 선택지에서 찾기 위한 단계별 계획을 세우세요."
                "결과에 <plan> tag를 붙여주세요."
                
                "주어진 문장:"
                "<paragraph>"
                "{paragraph}"

                "{question_plus}"
                "</paragraph>"

                "주어진 질문:"
                "<question>"
                "{question}"                                
                "</question>"

                "선택지:"
                "<choices>"
                "{list_choices}"
                "</choices>"
            )
        else:
            human = (
                "Paragraph을 참조하여 Question에 대한 적절한 답변을 List Choices 안에서 찾기 위한 단계별 계획을 세우세요."
                "결과에 <plan> tag를 붙여주세요."
                
                "Paragraph:"
                "{paragraph}"

                "{question_plus}"

                "Question:"
                "{question}"
                                                
                "List Choices:"
                "{list_choices}"
            )
                            
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )
        chat = get_chat()
        planner = planner_prompt | chat
        response = planner.invoke({
            "paragraph": paragraph,
            "question": question,
            "question_plus": question_plus,
            "list_choices": list_choices
        })
        print('response.content: ', response.content)
        result = response.content
        
        if not result.find('<plan>')==-1:
            output = result[result.find('<plan>')+6:result.find('</plan>')]
        else:
            output = result
        
        plan = output.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')
        print('planning_steps: ', planning_steps)

        notification = f"({idx}-{nth}) 생성된 계획:\n\n {planning_steps}"
        print('notification: ', notification)
        st.info(notification)
        
        return {
            "plan": planning_steps
        }

    def execute_node(state: State, config):
        print("###### execute ######")
        plan = state["plan"]
        # print('plan: ', plan) 
        
        list_choices = ""
        choices = state["choices"]
        for i, choice in enumerate(choices):
            list_choices += f"({i+1}) {choice}\n"
        # print('list_choices: ', list_choices)    
                
        idx = config.get("configurable", {}).get("idx")
        nth = config.get("configurable", {}).get("nth")

        notification = f"({idx}-{nth}) 실행중인 계획: {plan[0]}"
        print('notification: ', notification)
        st.info(notification)        
        
        task = plan[0]
        print('task: ', task)                        
        
        context = ""
        for info in state['info']:
            if isinstance(info, HumanMessage):
                context += info.content+"\n"
            else:
                context += info.content+"\n\n"
        # print('context: ', context)
                        
        system = (
            "당신은 국어 수능문제를 푸는 일타강사입니다."
            "매우 어려운 문제이므로 step by step으로 충분히 생각하고 답변합니다."
        )

        print('model_type: ', model_type)
        if model_type=="claude":
            human = (
                "당신의 목표는 <paragraph> tag의 주어진 문장으로 부터 <question> tag의 주어진 질문에 대한 적절한 답변을 <choice> tag의 선택지에서 찾는것입니다."
                "<previous_result> tag에 있는 이전 단계의 결과를 참조하여, <task> tag의 실행 단계를 수행하고 적절한 답변을 구합니다."
                "적절한 답변을 고를 수 없다면 다시 한번 읽어보고 가장 가까운 것을 선택합니다. 무조건 선택지중에 하나를 선택하여 답변합니다."
                "무조건 <choices> tag의 선택지 중에 하나를 선택하여 1-5 사이의 숫자로 답변합니다."
                "문제를 풀이할 때 모든 <choices> tag의 선택지에 항목마다 근거를 주어진 문장에서 찾아 설명하세요."
                "<choices> tag의 선택지의 주요 단어들의 의미를 주어진 문장과 비교해서 꼼꼼히 차이점을 찾습니다."
                "<choices> tag의 선택지를 모두 검토한 후에 최종 결과를 결정합니다."
                "최종 결과의 번호에 <result> tag를 붙여주세요."
                "최종 결과의 신뢰도를 1-5 사이의 숫자로 나타냅니다. 신뢰되는 <confidence> tag를 붙입니다."  
                                    
                "주어진 문장:"
                "<paragraph>"
                "{paragraph}"

                "{question_plus}"
                "</paragraph>"
                    
                "주어진 질문:"
                "<question>"
                "{question}"                    
                "</question>"
                
                "선택지:"
                "<choices>"
                "{list_choices}"
                "</choices>"
                
                "이전 단계의 결과:"
                "<previous_result>"
                "{info}"
                "</previous_result>"

                "실행 단계:"
                "<task>"
                "{task}"
                "</task>"
            )
        
        else:
            human = (
                "당신의 목표는 Paragraph으로 부터 Question에 대한 적절한 답변을 Question에서 찾는것입니다."
                "Past Results를 참조하여, Task를 수행하고 적절한 답변을 구합니다."
                "적절한 답변을 고를 수 없다면 다시 한번 읽어보고 가장 가까운 것을 선택합니다." 
                "받드시 List Choices중에 하나를 선택하여 1-5 사이의 숫자로 답변합니다."
                "문제를 풀이할 때 모든 List Choices마다 근거를 주어진 문장에서 찾아 설명하세요."
                "List Choices의 선택지의 주요 단어들의 의미를 Paragraph와 비교해서 자세히 차이점을 찾습니다."
                "List Choices의 선택지를 모두 검토한 후에 최종 결과를 결정합니다."                
                "최종 결과의 번호에 <result> tag를 붙여주세요."
                "최종 결과의 신뢰도를 1-5 사이의 숫자로 나타냅니다. 신뢰되는 <confidence> tag를 붙입니다."  

                "Past Results:"
                "{info}"
                                    
                "Paragraph:"
                "{paragraph}"

                "{question_plus}"
                    
                "Question:"
                "{question}"
                                
                "List Choices:"
                "{list_choices}"

                "Task:"
                "{task}"
            )
          
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )

        choice = confidence = 0
        result = ""
        for attempt in range(3):
            try:
                chat = get_chat()
                chain = prompt | chat                        
                response = chain.invoke({
                    "paragraph": state["paragraph"],
                    "question": state["question"],
                    "question_plus": state["question_plus"],
                    "list_choices": list_choices,
                    "info": context,
                    "task": task
                })
                print(f"attempt: {attempt}, response.content: {response.content}")

                idx = config.get("configurable", {}).get("idx")
                nth = config.get("configurable", {}).get("nth")

                notification = f"({idx}-{nth}) 실행된 결과입니다.\n{response.content}"
                print('notification: ', notification)
                st.info(notification)   
            
                result = response.content
                if not result.find('<confidence>')==-1:
                    output = result[result.find('<confidence>')+12:result.find('</confidence>')]
                    print('output: ', output)
                    confidence = string_to_int(output)
                    print('confidence: ', confidence)
                if not result.find('<result>')==-1:
                    output = result[result.find('<result>')+8:result.find('</result>')]
                    print('output: ', output)
                    choice = string_to_int(output)
                    print('choice: ', choice)
                break
            except Exception:
                response = AIMessage(content="답변을 찾지 못하였습니다.")

                err_msg = traceback.format_exc()
                print('error message: ', err_msg)
        
        transaction = [HumanMessage(content=task), AIMessage(content=result)]
        # print('transaction: ', transaction)
        
        if confidence >= 4 and choice>0 and choice<6:
            plan = []            
            answer = choice
            
        else:
            plan = state["plan"]
            answer = 0
        
        return {
            "plan": plan,
            "info": transaction,
            "past_steps": [task],
            "answer": answer
        }

    def replan_node(state: State, config):
        print('#### replan ####')
        # print('state of replan node: ', state)        
        print('past_steps: ', state["past_steps"])
                
        idx = config.get("configurable", {}).get("idx")
        nth = config.get("configurable", {}).get("nth")
        
        if len(state["plan"])==0:
            return {"plan": []}
        
        notification = f"({idx}-{nth}) 새로운 계획을 생성합니다..."
        print('notification: ', notification)
        st.info(notification)
        
        system = (
            "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
        )        

        if model_type=="claude":
            human = (
                "<paragraph> tag의 주어진 문장과 <question> tag의 주어진 질문 참조하여 <choices> tag의 선택지에서 거장 적절한 항목을 선택하기 위해서는 잘 세워진 계획이 있어야 합니다."                
                "<original_plan> tag의 원래 계획을 상황에 맞게 수정하세요."
                "<paragraph> tag의 주어진 문장과 <question> tag의 주어진 질문을 참조하여 선택지에서 거장 적절한 항목을 선택하기 위해서는 잘 세워진 계획이 있어야 합니다."
                "<original_plan> tag에 있는 당신의 원래 계획에서 아직 수행되지 않은 계획들을 수정된 계획에 포함하세요."
                "수정된 계획에는 <past_steps> tag의 완료한 단계는 포함하지 마세요."
                "새로운 계획에는 <plan> tag를 붙여주세요."

                "주어진 문장:"
                "<paragraph>"
                "{paragraph}"

                "{question_plus}"
                "</paragraph>"
                
                "주어진 질문:"
                "<question>"
                "{question}"
                "</question>"
                
                "선택지:"
                "<choices>"
                "{list_choices}"
                "</choices>"
                
                "원래 계획:" 
                "<original_plan>"                
                "{plan}"
                "</original_plan>"

                "완료한 단계:"
                "<past_steps>"
                "{past_steps}"
                "</past_steps>"                                
                
                "수정된 계획의 형식은 아래와 같습니다."
                "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
                "1. [질문을 해결하기 위한 단계]"
                "2. [질문을 해결하기 위한 단계]"
                "..."         
            )
        else:
            human = (
                "Paragraph과 Question을 참조하여 List Choices에서 거장 적절한 항목을 선택하기 위해서는 잘 세워진 계획이 있어야 합니다."
                "Original Steps를 상황에 맞게 수정하여 새로운 계획을 세우세요."
                "Original Steps의 첫번째 단계는 이미 완료되었으니 절대 포함하지 않습니다."
                "Original Plan에서 아직 수행되지 않은 단계를 새로운 계획에 포함하세요."
                "Past Steps의 완료한 단계는 계획에 포함하지 마세요."
                "새로운 계획에는 <plan> tag를 붙여주세요."
                
                "Paragraph:"
                "{paragraph}"

                "{question_plus}"
                
                "Question:"
                "{question}"
                
                "List Choices:"
                "{list_choices}"
                
                "Original Steps:" 
                "{plan}"
                
                "Past Steps:"
                "{past_steps}"
                
                "새로운 계획의 형식은 아래와 같습니다."
                "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
                "1. [질문을 해결하기 위한 단계]"
                "2. [질문을 해결하기 위한 단계]"
                "..."         
            )                    
        replanner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )        
        chat = get_chat()
        replanner = replanner_prompt | chat        


        # plans = '\n'.join(state["plan"])
        # print('plans: ', plans)
        # past_steps = '\n'.join(state["past_steps"])
        # print('past_steps: ', past_steps)
        list_choices = ""
        choices = state["choices"]
        for i, choice in enumerate(choices):
            list_choices += f"({i+1}) {choice}\n\n"
        print('list_choices: ', list_choices)    

        response = replanner.invoke({
            "paragraph": state["paragraph"],
            "question_plus": state["question_plus"],
            "question": state["question"],
            "list_choices": list_choices,
            "plan": state["plan"],
            "past_steps": state["past_steps"]
        })
        print('response.content: ', response.content)
        result = response.content        
        
        if result.find('<plan>') == -1:
            print('result: ', result)
            st.info(result)
            
            return {"plan":[]}
        else:
            output = result[result.find('<plan>')+6:result.find('</plan>')]
            print('plan output: ', output)
            
            plans = output.strip().replace('\n\n', '\n')
            planning_steps = plans.split('\n')
            print('planning_steps: ', planning_steps)

            notification = f"({idx}-{nth}) 생성된 계획:\n\n {planning_steps}"
            print('notification: ', notification)
            st.info(notification)
        
            return {"plan": planning_steps}
                
    def should_end(state: State) -> Literal["continue", "end"]:
        print('#### should_end ####')
        # print('state: ', state)
        
        plan = state["plan"]
        print('plan: ', plan)
        if len(plan)<=1:
            next = "end"
        else:
            next = "continue"
        print(f"should_end response: {next}")
        
        return next
        
    def final_answer(state: State, config) -> str:
        print('#### final_answer ####')
        
        idx = config.get("configurable", {}).get("idx")
        nth = config.get("configurable", {}).get("nth")
        correct_answer = config.get("configurable", {}).get("correct_answer")
        score = config.get("configurable", {}).get("score")
        
        notification = f"({idx}-{nth}) 최종 답변을 구합니다..."
        print('notification: ', notification)
        st.info(notification)
                
        answer = state["answer"]

        print(f'answer: {answer}, correct_answer: {correct_answer}')
        print(f'Type--> answer: {answer.__class__}, correct_answer: {correct_answer.__class__}')
        if len(state["plan"])==0:
            # for debuuging
            if answer == correct_answer:
                st.warning(f"({idx}-{nth}) 정답입니다. (score: {score})")
            else:
                st.error(f"({idx}-{nth}) 오답입니다. 정답은 {correct_answer}입니다. (score: {score})")        
            
            # return final answer
            return {"answer": answer}
        
        # get final answer
        info = state['info']
        
        context = ""
        for info in state['info']:
            if isinstance(info, HumanMessage):
                context += info.content+"\n"
            else:
                context += info.content+"\n\n"
        print('context: ', context)
                                
        print('paragraph: ', state["paragraph"])
        print('question: ', state["question"])
        print('question_plus: ', state["question_plus"])
        
        list_choices = ""
        choices = state["choices"]
        for i, choice in enumerate(choices):
            list_choices += f"({i+1}) {choice}\n"
        print('list_choices: ', list_choices)
        
        system = (
            "당신은 국어 수능문제를 푸는 일타강사입니다."
            "매우 어려운 문제이므로 step by step으로 충분히 생각하고 답변합니다."
        )    

        if model_type=="claude":    
            human = (
                "<context> tag에 있는 검토 결과를 활용하여, <paragraph> tag의 주어진 문장으로 부터 <question> tag의 주어진 질문에 대한 적절한 답변을 <choice> tag의 선택지 안에서 선택하려고 합니다."
                "가장 가까운 선택지를 골라서 반드시 번호로 답변 합니다."
                "답변을 고를 수 없다면 다시 한번 읽어보고 가장 가까운 항목을 선택합니다. 무조건 선택지중에 하나를 선택하여 답변합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "최종 결과의 번호에 <result> tag를 붙여주세요."
                "최종 결과의 신뢰도를 1-5 사이의 숫자로 나타냅니다. 신뢰되는 <confidence> tag를 붙입니다."  
                
                "이전 단계에서 검토한 결과:"
                "<context>"
                "{context}"
                "</context>"
                                
                "주어진 문장:"
                "<paragraph>"
                "{paragraph}"
                
                "{question_plus}"                
                "</paragraph>"

                "주어진 질문:"
                "<question>"
                "{question}"
                "</question>"

                "선택지:"
                "<choices>"
                "{list_choices}"
                "</choices>"       
            )
        else:
            human = (
                "당신은 Past Results를 활용하여, Paragraph으로 부터 Question에 대한 적절한 답변을 List Choices 안에서 선택하려고 합니다."
                "가장 가까운 List Choices를 골라서 반드시 번호로 답변 합니다."
                "답변을 고를 수 없다면 다시 한번 읽어보고 가장 가까운 항목을 선택합니다. 무조건 List Choices중에 하나를 선택하여 답변합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "Paragraph의 사실관계를 파악하여 전체적으로 읽어가면서 Question을 이해합니다."
                "최종 결과 번호에 <result> tag를 붙여주세요. 예) <result>1</result>"  
                "최종 결과의 신뢰도를 1-5 사이의 숫자로 나타냅니다. 신뢰되는 <confidence> tag를 붙입니다."  
                
                "Past Results:"
                "{context}"
                                
                "Paragraph:"
                "{paragraph}"

                "{question_plus}"

                "Question:"
                "{question}"
                
                "List Choices:"
                "{list_choices}"
            )
                
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

        answer = 0
        for attempt in range(3):
            try:
                chat = get_chat()
                chain = prompt | chat                
                response = chain.invoke(
                    {
                        "context": context,
                        "paragraph": state["paragraph"],                    
                        "question": state["question"],
                        "question_plus": state["question_plus"],
                        "list_choices": list_choices
                    }
                )
                result = response.content
                print(f"attempt: {attempt}, result: {result}")

                notification = f"({idx}-{nth}) 최종으로 얻어진 결과:\n\n{result}"
                print('notification: ', notification)
                st.info(notification)

                if not result.find('<confidence>')==-1:
                    output = result[result.find('<confidence>')+12:result.find('</confidence>')]
                    print('output: ', output)

                    confidence = string_to_int(output)
                    print('confidence: ', confidence)

                if not result.find('<result>')==-1:
                    output = result[result.find('<result>')+8:result.find('</result>')]
                    print('output: ', output)
                    answer = string_to_int(output)
                    print('answer: ', answer)
                break
            except Exception:
                    response = AIMessage(content="답변을 찾지 못하였습니다.")
                    err_msg = traceback.format_exc()
                    print('error message: ', err_msg)

        print(f'answer: {answer}, correct_answer: {correct_answer}')
        print(f'Type--> answer: {answer.__class__}, correct_answer: {correct_answer.__class__}')
        if answer == correct_answer:
            st.warning(f"({idx}-{nth}) 정답입니다. (score: {score})")
        else:
            st.error(f"({idx}-{nth}) 오답입니다. 정답은 {correct_answer}입니다. (score: {score})")

        return {"answer":answer}  

    def buildPlanAndExecute():
        workflow = StateGraph(State)
        workflow.add_node("planner", plan_node)
        workflow.add_node("executor", execute_node)
        workflow.add_node("replaner", replan_node)
        workflow.add_node("final_answer", final_answer)
        
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "replaner")
        workflow.add_conditional_edges(
            "replaner",
            should_end,
            {
                "continue": "executor",
                "end": "final_answer",
            },
        )
        workflow.add_edge("final_answer", END)

        return workflow.compile()

    # run graph   
    app = buildPlanAndExecute()    
    
    inputs = {
        "question": question,
        "question_plus": question_plus,
        "paragraph": paragraph,
        "choices": choices
    }
    config = {
        "idx": idx,
        "nth": nth,
        "correct_answer": correct_answer,
        "score": score,
        "recursion_limit": 50
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)            
    print('value: ', value)    

    answer = value["answer"]
    print('final answer: ', answer)
    
    notification = f"({idx}-{nth}) 최종 답변은 {answer}입니다."
    print('notification: ', notification)
    st.info(notification)
        
    return answer

####################### LangChain #######################
# Translation (English)
#########################################################

def translate_text(text, model_name):
    global llmMode
    llmMode = model_name

    chat = get_chat()

    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
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
        print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    if msg.find('<result>') != -1:
        msg = msg[msg.find('<result>')+8:msg.find('</result>')] # remove <result> tag
    if msg.find('<article>') != -1:
        msg = msg[msg.find('<article>')+9:msg.find('</article>')] # remove <article> tag

    return msg


####################### LangChain #######################
# Image Summarization
#########################################################

def get_image_summarization(object_name, prompt, st):
    # load image
    s3_client = boto3.client(
        service_name='s3',
        region_name=bedrock_region
    )

    if debug_mode=="Enable":
        status = "이미지를 가져옵니다."
        print('status: ', status)
        st.info(status)
                
    image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+object_name)
    # print('image_obj: ', image_obj)
    
    image_content = image_obj['Body'].read()
    img = Image.open(BytesIO(image_content))
    
    width, height = img.size 
    print(f"width: {width}, height: {height}, size: {width*height}")
    
    isResized = False
    while(width*height > 5242880):                    
        width = int(width/2)
        height = int(height/2)
        isResized = True
        print(f"width: {width}, height: {height}, size: {width*height}")
    
    if isResized:
        img = img.resize((width, height))
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # extract text from the image
    if debug_mode=="Enable":
        status = "이미지에서 텍스트를 추출합니다."
        print('status: ', status)
        st.info(status)

    text = extract_text(img_base64)
    print('extracted text: ', text)

    if text.find('<result>') != -1:
        extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
        # print('extracted_text: ', extracted_text)
    else:
        extracted_text = text
    
    if debug_mode=="Enable":
        status = f"### 추출된 텍스트\n\n{extracted_text}"
        print('status: ', status)
        st.info(status)
    
    if debug_mode=="Enable":
        status = "이미지의 내용을 분석합니다."
        print('status: ', status)
        st.info(status)

    image_summary = summary_image(img_base64, prompt)
    
    if text.find('<result>') != -1:
        image_summary = image_summary[image_summary.find('<result>')+8:image_summary.find('</result>')]
    print('image summary: ', image_summary)
            
    if len(extracted_text) > 10:
        contents = f"## 이미지 분석\n\n{image_summary}\n\n## 추출된 텍스트\n\n{extracted_text}"
    else:
        contents = f"## 이미지 분석\n\n{image_summary}"
    print('image contents: ', contents)

    return contents

