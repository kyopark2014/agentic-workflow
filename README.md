# Nova Pro로 Agentic Workflow 활용하기

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Flanggraph-nova&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false")](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>


여기서는 LangGraph로 구현한 agentic workflow를 구현하고 [Streamlit](https://streamlit.io/)을 이용해 개발 및 테스트 환경을 제공합니다. 한번에 배포하고 바로 활용할 수 있도록 [AWS CDK](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-construct-library.html)를 이용하고 ALB - EC2의 구조를 이용해 scale out도 구현할 수 있습니다. 또한, [CloudFront](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Introduction.html) - ALB 구조를 이용해 배포후 바로 HTTPS로 접속할 수 있습니다. Agentic workflow는 tool use, reflection, planning, multi-agent collaboration을 구현하고 있습니다.

## System Architecture 

이때의 architecture는 아래와 같습니다. 여기서에서는 streamlit이 설치된 EC2는 private subnet에 둬서 안전하게 관리합니다. RAG는 Knowledge base를 이용해 손쉽게 동기화 및 문서관리가 가능합니다. 이때 Knowledge base의 data source로는 OpenSearch를 활용하고 있습니다. 인터넷 검색은 tavily를 사용하고 날씨 API를 추가로 활용합니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/ac1dc2a8-3bb8-4d77-8d15-cf708e3541ab" />


## 상세 구현

Agentic workflow (tool use)는 아래와 같이 구현할 수 있습니다. 상세한 내용은 [chat.py](./application/chat.py)을 참조합니다.

### Basic Chat

일반적인 대화는 아래와 같이 stream으로 결과를 얻을 수 있습니다. 여기에서는 LangChain의 ChatBedrock과 Nova Pro의 모델명인 "us.amazon.nova-pro-v1:0"을 활용하고 있습니다.

```python
modelId = "us.amazon.nova-pro-v1:0"
bedrock_region = "us-west-2"
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
    "stop_sequences": ["\n\n<thinking>", "\n<thinking>", " <thinking>"]
}

chat = ChatBedrock(  
    model_id=modelId,
    client=boto3_bedrock, 
    model_kwargs=parameters,
    region_name=bedrock_region
)

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
stream = chain.stream(
    {
        "history": history,
        "input": query,
    }
)  
print('stream: ', stream)
```

### RAG

여기에서는 RAG 구현을 위하여 Amazon Bedrock의 knowledge base를 이용합니다. Amazon S3에 필요한 문서를 올려놓고, knowledge base에서 동기화를 하면, OpenSearch에 문서들이 chunk 단위로 저장되므로 문서를 쉽게 RAG로 올리고 편하게 사용할 수 있습니다. 또한 Hiearchical chunk을 위하여 검색 정확도를 높이면서 필요한 context를 충분히 제공합니다. 


LangChain의 [AmazonKnowledgeBasesRetriever](https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.bedrock.AmazonKnowledgeBasesRetriever.html)을 이용하여 retriever를 등록합니다. 

```python
from langchain_aws import AmazonKnowledgeBasesRetriever

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=knowledge_base_id, 
    retrieval_config={"vectorSearchConfiguration": {
        "numberOfResults": top_k,
        "overrideSearchType": "HYBRID"   
    }},
    region_name=bedrock_region
)
```

Knowledge base로 조회하여 얻어진 문서를 필요에 따라 아래와 같이 재정리합니다. 이때 파일 경로로 사용하는 url은 application에서 다운로드 가능하도록 CloudFront의 도메인과 파일명을 조화합여 생성합니다.

```python
documents = retriever.invoke(query)
for doc in documents:
    content = ""
    if doc.page_content:
        content = doc.page_content    
    score = doc.metadata["score"]    
    link = ""
    if "s3Location" in doc.metadata["location"]:
        link = doc.metadata["location"]["s3Location"]["uri"] if doc.metadata["location"]["s3Location"]["uri"] is not None else ""        
        pos = link.find(f"/{doc_prefix}")
        name = link[pos+len(doc_prefix)+1:]
        encoded_name = parse.quote(name)
        link = f"{path}{doc_prefix}{encoded_name}"        
    elif "webLocation" in doc.metadata["location"]:
        link = doc.metadata["location"]["webLocation"]["url"] if doc.metadata["location"]["webLocation"]["url"] is not None else ""
        name = "WEB"
    url = link
            
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
```        

얻어온 문서가 적절한지를 판단하기 위하여 아래와 같이 prompt를 이용해 관련도를 평가하고 [structured output](https://github.com/kyopark2014/langgraph-agent/blob/main/structured-output.md)을 이용해 결과를 추출합니다.

```python
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

filtered_docs = []
for i, doc in enumerate(documents):
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                
    grade = score.binary_score
    if grade.lower() == "yes":
        print("---GRADE: DOCUMENT RELEVANT---")
        filtered_docs.append(doc)
    else:
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        continue
```

이후 아래와 같이 RAG를 활용하여 원하는 응답을 얻습니다.

```python
system = (
  "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
  "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
  "모르는 질문을 받으면 솔직히 모른다고 말합니다."
  "답변의 이유를 풀어서 명확하게 설명합니다."
)
human = (
    "Question: {input}"

    "Reference texts: "
    "{context}"
)    
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
chain = prompt | chat
stream = chain.invoke(
    {
        "context": context,
        "input": revised_question,
    }
)
print(stream.content)    
```

### Agentic Workflow: Tool Use

아래와 같이 activity diagram을 이용하여 node/edge/conditional edge로 구성되는 tool use 방식의 agent를 구현할 수 있습니다.

<img width="261" alt="image" src="https://github.com/user-attachments/assets/31202a6a-950f-44d6-b50e-644d28012d8f" />

Tool use 방식 agent의 workflow는 아래와 같습니다. Fuction을 선택하는 call model 노드과 실행하는 tool 노드로 구성됩니다. 선택된 tool의 결과에 따라 cycle형태로 추가 실행을 하거나 종료하면서 결과를 전달할 수 있습니다.

```python
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

app = workflow.compile()
inputs = [HumanMessage(content=query)]
config = {
    "recursion_limit": 50
}
message = app.invoke({"messages": inputs}, config)
```

Tool use 패턴의 agent는 정의된 tool 함수의 docstring을 이용해 목적에 맞는 tool을 선택합니다. 아래의 search_by_knowledge_base는 OpenSearch를 데이터 저장소로 사용하는 knowledbe base로 부터 관련된 문서를 얻어오는 tool의 예입니다. "Search technical information by keyword"로 정의하였으므로 질문이 기술적인 내용이라면 search_by_knowledge_base가 호출되게 됩니다.

```python
@tool    
def search_by_knowledge_base(keyword: str) -> str:
    """
    Search technical information by keyword and then return the result as a string.
    keyword: search keyword
    return: the technical information of keyword
    """    
    
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID" 
            }},
        )        
        docs = retriever.invoke(keyword)
    
    relevant_context = ""
    for i, doc in enumerate(docs):
        relevant_context += doc.page_content + "\n\n"        
    return relevant_context    
```



아래와 같이 tool들로 tools를 정의한 후에 [bind_tools](https://python.langchain.com/docs/how_to/chat_models_universal_init/#using-a-configurable-model-declaratively)을 이용하여 call_model 노드를 정의합니다. 

```python
tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_knowledge_base]        

def call_model(state: State, config):
    system = (
        "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    model = chat.bind_tools(tools)
    chain = prompt | model
        
    response = chain.invoke(state["messages"])

    return {"messages": [response]}
```

또한, tool 노드는 아래와 같이 [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode)을 이용해 정의합니다.

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```


### Agentic Workflow: Reflection

Reflection은 generate, reflect, revise의 과정을 통해 초안(draft)을 향상시킵니다.

<img width="205" alt="image" src="https://github.com/user-attachments/assets/5a9b547f-afc8-427e-9172-9dc5648ec512" />

아래와 같이 generate, reflect, revise_answer로 된 노드를 구성하고 conditional edge인 should_continue()을 통해 max_revision만큼 반복합니다.

```python
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

inputs = [HumanMessage(content=query)]
config = {
    "recursion_limit": 50
}
message = app.invoke({"messages": inputs}, config)
print(event["messages"][-1].content)
```

Reflection에서는 부족(missing), 조언(advisable), 추가 검색어(search_queries)를 추출합니다. 아래와 같이 structued_output을 활용합니다.

```python
class Reflection(BaseModel):
    missing: str = Field(description="작성된 글에 있어야하는데 빠진 내용이나 단점")
    advisable: str = Field(description="더 좋은 글이 되기 위해 추가하여야 할 내용")
    superfluous: str = Field(description="글의 길이나 스타일에 대한 비평")

class Research(BaseModel):
    """글쓰기를 개선하기 위한 검색 쿼리를 제공합니다."""
    reflection: ReflectionKor = Field(description="작성된 글에 대한 평가")
    search_queries: list[str] = Field(
        description="도출된 비평을 해결하기 위한 3개 이내의 검색어"            
    )    

def reflect(state: State, config):
    reflection = []
    search_queries = []

    structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)    
    info = structured_llm.invoke(state["messages"][-1].content)        
    if not info['parsed'] == None:
        parsed_info = info['parsed']
        reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
        search_queries = parsed_info.search_queries
    return {
        "messages": state["messages"],
        "reflection": reflection,
        "search_queries": search_queries
    }
```



### Agentic Workflow: Planning

Planning 패턴에서는 plan, execute, reflan을 통해 논리적인 글을 쓰거나 복잡한 문제를 해결할 수 있습니다.

<img width="98" alt="image" src="https://github.com/user-attachments/assets/58aa8302-cb56-45e3-b225-f6c9c7a6ab66" />

아래와 같이 planning을 위한 workflow를 정의합니다.

```python
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
app = workflow.compile()
```

여기서 planning을 아래와 같이 생성하고 executor로 실행합니다.

```python
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
result = response.content
plan = result.strip().replace('\n\n', '\n')
planning_steps = plan.split('\n')
```

replan은 아래와 같이 수행할 수 있습니다.

```python
system = (
    "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
    "당신은 다음의 Question에 대한 적절한 답변을 얻고자합니다."
)        
human = (
    "Question: {input}"
                
    "당신의 원래 계획은 아래와 같습니다." 
    "Original Plan:"
    "{plan}"

    "완료한 단계는 아래와 같습니다."
    "Past steps:"
    "{past_steps}"
    
    "당신은 Original Plan의 원래 계획을 상황에 맞게 수정하세요."
    "계획에 아직 해야 할 단계만 추가하세요. 이전에 완료한 단계는 계획에 포함하지 마세요."                
    "수정된 계획에는 <plan> tag를 붙여주세요."
    "만약 더 이상 계획을 세우지 않아도 Question의 주어진 질문에 답변할 있다면, 최종 결과로 Question에 대한 답변을 <result> tag를 붙여 전달합니다."
    
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
replanner = replanner_prompt | chat

response = replanner.invoke({
    "input": state["input"],
    "plan": state["plan"],
    "past_steps": state["past_steps"]
})
print('replanner output: ', response.content)
result = response.content

plans = result.strip().replace('\n\n', '\n')
planning_steps = plans.split('\n')
```        

### Agentic Workflow: Multi-agent Collaboration

<img width="416" alt="image" src="https://github.com/user-attachments/assets/3d8e0527-cbf3-4d72-855b-4067150e3b19" />

여기서 설명하는 multi-agent collaboration은 planning과 reflection을 수행하는 agent를 이용하여 구현됩니다. 먼저 planning agent의 workflow는 아래와 같이 구성할 수 있습니다. 

```python
workflow = StateGraph(State)

# Add nodes
workflow.add_node("planning_node", plan_node)
workflow.add_node("execute_node", execute_node)
workflow.add_node("revising_node", revise_answer)

# Set entry point
workflow.set_entry_point("planning_node")

# Add edges
workflow.add_edge("planning_node", "execute_node")
workflow.add_edge("execute_node", "revising_node")
workflow.add_edge("revising_node", END)

planning_app = workflow.compile()
```

planning에서 생성된 draft를 개선하기 위하여 reflection agent를 아래와 같이 정의하여 활용할 수 있습니다.

```python
workflow = StateGraph(ReflectionState)

# Add nodes
workflow.add_node("reflect_node", reflect_node)
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

reflection_app = workflow.compile()
```
    


### 활용 방법

EC2는 Private Subnet에 있으므로 SSL로 접속할 수 없습니다. 따라서, [Console-EC2](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)에 접속하여 "app-for-llm-streamlit"를 선택한 후에 Connect에서 sesseion manager를 선택하여 접속합니다. 

Github에서 app에 대한 코드를 업데이트 하였다면, session manager에 접속하여 아래 명령어로 업데이트 합니다. 

```text
sudo runuser -l ec2-user -c 'cd /home/ec2-user/langgraph-nova && git pull'
```

Streamlit의 재시작이 필요하다면 아래 명령어로 service를 stop/start 시키고 동작을 확인할 수 있습니다.

```text
sudo systemctl stop streamlit
sudo systemctl start streamlit
sudo systemctl status streamlit -l
```

Local에서 디버깅을 빠르게 진행하고 싶다면 [Local에서 실행하기](https://github.com/kyopark2014/llm-streamlit/blob/main/deployment.md#local%EC%97%90%EC%84%9C-%EC%8B%A4%ED%96%89%ED%95%98%EA%B8%B0)에 따라서 Local에 필요한 패키지와 환경변수를 업데이트 합니다. 이후 아래 명령어서 실행합니다.

```text
streamlit run application/app.py
```

EC2에서 debug을 하면서 개발할때 사용하는 명령어입니다.

먼저, 시스템에 등록된 streamlit을 종료합니다.

```text
sudo systemctl stop streamlit
```

이후 EC2를 session manager를 이용해 접속한 이후에 아래 명령어를 이용해 실행하면 로그를 보면서 수정을 할 수 있습니다. 

```text
sudo runuser -l ec2-user -c "/home/ec2-user/.local/bin/streamlit run /home/ec2-user/langgraph-nova/application/app.py"
```



## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 

## 실행 결과

### RAG (Knowledge Base)

메뉴에서 RAG를 선택하고, "AWS의 스토리지 서비스에 대해 설명해주세요."라고 입력 후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/89dbe5f3-0dd1-4829-af80-a5fc51ad03e7)

이번에는 "Bedrock Agent와 S3를 비교해 주세요" 라고 입력후에 결과를 확인합니다. RAG만 적용한 경우에는 사용자의 질문을 그대로 검색하는데, 정확히 관련된 문서가 없으면 적절히 답변할 수 없습니다. 

![image](https://github.com/user-attachments/assets/a365357a-aaec-4745-ab74-fc3bcb769873)

메뉴에서 RAG를 선택하고 "Amazon Nova Pro 모델에 대해 설명해주세요"라고 입력하고 결과를 확인하면 아래와 같습니다. RAG는 retrieve - grade - generate의 단계를 통해 질문에 대한 답변 및 관련 문서를 제공합니다. RAG의 경우에 query decomposition을 하지 못하므로 입력된 질문을 Knowledge base로 구현된 RAG에 직접 질문하게 됩니다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/f84e4fa2-c7b8-492f-8c0c-82b32cb1426b" />


### Agentic Workflow

Agentic Workflow(Tool Use) 메뉴를 선택하여 오늘 서울의 날씨에 대해 질문을 하면, 아래와 같이 입력하고 결과를 확인합니다. LangGraph로 구현된 Tool Use 패턴의 agent는 날씨에 대한 요청이 올 경우에 openweathermap의 API를 요청해 날씨정보를 조회하여 활용할 수 있습니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/4693c1ff-b7e9-43f5-b7b7-af354b572f07" />

아래와 같은 질문은 LLM이 가지고 있지 않은 정보이므로, 인터넷 검색을 수행하고 그 결과로 아래와 같은 답변을 얻었습니다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/8f8d2e94-8be1-4b75-8795-4db9a8fa340f" />


RAG를 테스트 하였을때에 사용한 "Bedrock Agent와 S3를 비교해 주세요."라고 질문을 하면, 이번에는 좀더 나은 답변을 얻었습니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/969e9b84-5b80-4948-8627-f86bd2af26bc" />


메뉴에서 multi-agent collaboration을 선택한 후에 "지방 조직이 분비하는 exosome들이 어떻게 면역체계에 역할을 하고 어떻게 하면 좋은 exosome들을 분비시켜 당뇨나 병을 예방할수 있는지 알려주세요."라고 입력합니다. 이때 생성된 결과는 아래와 같습니다.

[지방 조직이 분비하는 exosome들이 어떻게 면역체계에 역할을 하고 어떻게 하면 좋은 exosome들을 분비시켜 당뇨나 병을 예방할수 있는지 알려주세요.](https://github.com/kyopark2014/langgraph-nova/blob/main/contents/%EC%A7%80%EB%B0%A9_%EC%A1%B0%EC%A7%81_exosome%EC%9D%98_%EB%A9%B4%EC%97%AD_%EC%97%AD%ED%95%A0_%EB%B0%8F_%EC%98%88%EB%B0%A9_%EB%B0%A9%EB%B2%95.md)



### Reference 

[Nova Pro User Guide](https://docs.aws.amazon.com/pdfs/nova/latest/userguide/nova-ug.pdf)
