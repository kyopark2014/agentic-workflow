# Agentic Workflow 활용하기

<p align="left">
    <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square">
</p>


여기서는 오픈소스 LLM Framework인 [LangGraph](https://langchain-ai.github.io/langgraph/)을 이용하여 tool use, reflection, planning, multi-agent collaboration 방식으로 workflow를 수행하는 agent를 구현합니다. 구현된 workflow들은 [Streamlit](https://streamlit.io/)을 이용해 개발 및 테스트를 수행할 수 있습니다. [AWS CDK](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-construct-library.html)를 이용하고 한번에 배포할 수 있고, [CloudFront](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Introduction.html) - ALB 구조를 이용해 HTTPS로 안전하게 접속할 수 있습니다. 

## System Architecture 

전체적인 architecture는 아래와 같습니다. Streamlit이 설치된 EC2는 private subnet에 있고, CloudFront-ALB를 이용해 외부와 연결됩니다. RAG는 Knowledge base를 이용해 손쉽게 동기화 및 문서관리가 가능합니다. 이때 Knowledge base의 data source로는 OpenSearch를 활용하고 있습니다. 인터넷 검색은 tavily를 사용하고 날씨 API를 추가로 활용합니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/9d48c4d0-554b-4211-812b-329571f36b97" />

## 상세 구현

Agentic workflow (tool use)는 아래와 같이 구현할 수 있습니다. 상세한 내용은 [chat.py](./application/chat.py)을 참조합니다.

### Basic Chat

일반적인 대화는 아래와 같이 stream으로 결과를 얻을 수 있습니다. 여기에서는 LangChain의 ChatBedrock과 Nova Pro의 모델명인 "us.amazon.nova-pro-v1:0"을 활용하고 있습니다.

```python
bedrock_region = "us-west-2"
modelId = "us.amazon.nova-pro-v1:0"
stop_sequences = ["\n\n<thinking>", "\n<thinking>", " <thinking>"]

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
    "stop_sequences": stop_sequences
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

여기에서는 RAG 구현을 위하여 Amazon Bedrock의 knowledge base를 이용합니다. Amazon S3에 필요한 문서를 올려놓고, knowledge base에서 동기화를 하면, OpenSearch에 문서들이 chunk 단위로 저장되므로 문서를 쉽게 RAG로 올리고 편하게 사용할 수 있습니다. 또한 Hiearchical chunk을 이용하여 검색 정확도를 높이면서 필요한 context를 충분히 제공합니다. 


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

<img width="300" alt="image" src="https://github.com/user-attachments/assets/68365969-73ef-40ed-a306-1a1d147dfd4e" />

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

<img width="200" alt="image" src="https://github.com/user-attachments/assets/cee8559f-d2c9-4d7a-82a9-83772dba086f" />

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

### Agentic Workflow: Multi-agent Collaboration (Deep Research Agent)

여기서 설명하는 multi-agent collaboration은 planning과 reflection을 수행하는 agent를 이용하여 구현됩니다. 

<img src="https://github.com/user-attachments/assets/486d748d-cdf4-4e52-ab9a-785116176198" width="800">

먼저 planning agent의 workflow는 아래와 같이 구성할 수 있습니다. 

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

## Code Interpreter

여기에서는 Agent의 Tool로 Code Interpreter를 활용하는 방법에 대해 설명합니다. Code Interpreter를 이용하면, 언어모델에서 어려운 복잡한 계산이나 그래프를 그리는 일들을 수행할 수 있습니다. 


### Python REPL 

LangChain에서 제공하는 Python REPL (read-eval-print loop)을 이용하면 Python 코드를 실행할 수 있습니다. 

[PythonAstREPLTool](https://python.langchain.com/api_reference/experimental/tools/langchain_experimental.tools.python.tool.PythonAstREPLTool.html#langchain_experimental.tools.python.tool.PythonAstREPLTool)을 이용해 구현합니다. PythonAstREPLTool을 활용하기 위해서 langchain_experimental을 아래와 같이 설치합니다.

```text
pip install langchain_experimental
```

code를 실행하기 위해 repl_coder를 정의합니다. 

```python
from langchain_experimental.tools import PythonAstREPLTool
repl = PythonAstREPLTool()

@tool
def repl_coder(code):
    """
    Use this to execute python code and do math. 
    If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.
    code: the Python code was written in English
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    if result is None:
        result = "It didn't return anything."

    return result
```

그래프를 그리기 위한 라이브러리를 설치합니다.

```text
pip install numpy matplotlib
```

그래프는 stdout으로 받아서 이미지로 저장후 활용합니다.

```python
@tool
def repl_drawer(code):
    """
    Execute a Python script for draw a graph.
    Since Python runtime cannot use external APIs, necessary data must be included in the code.
    The graph should use English exclusively for all textual elements.
    Do not save pictures locally bacause the runtime does not have filesystem.
    When a comparison is made, all arrays must be of the same length.
    code: the Python code was written in English
    return: the url of graph
    """ 
        
    code = re.sub(r"seaborn", "classic", code)
    code = re.sub(r"plt.savefig", "#plt.savefig", code)
    code = re.sub(r"plt.show", "#plt.show", code)

    post = """\n
import io
import base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.getvalue()).decode()
print(image_base64)
"""
    code = code + post

    result = ""
    resp = repl.run(code)
    base64Img = resp
    
    if base64Img:
        byteImage = BytesIO(base64.b64decode(base64Img))

        image_name = generate_short_uuid()+'.png'
        url = chat.upload_to_s3(byteImage, image_name)

        file_name = url[url.rfind('/')+1:]

        global image_url
        image_url.append(path+'/'+s3_image_prefix+'/'+parse.quote(file_name))
        result = f"생성된 그래프의 URL: {image_url}"

    return result
```

아래와 같이 repl_coder와 repl_drawer는 tool에 등록해서 agent에서 활용합니다.

```python
tools = [repl_coder, repl_drawer]
```


### Sandbox에서 실행하기

Sandbox 환경에서 Code Interpreter를 이용할 때에는 직접 container로 환경을 만들거나, [Jupyter Kernel Gateway](https://github.com/jupyter-server/kernel_gateway)을 이용하는 방안은 검토할 수 있으나 구현의 복잡성 등으로 인해서, [Riza](https://docs.riza.io/introduction)나 [E2B](https://www.linkedin.com/feed/update/urn:li:activity:7191459920251109377/?commentUrn=urn%3Ali%3Acomment%3A(activity%3A7191459920251109377%2C7295624350970363904)&dashCommentUrn=urn%3Ali%3Afsd_comment%3A(7295624350970363904%2Curn%3Ali%3Aactivity%3A7191459920251109377))와 같은 API를 활용할 수 있고, 아래에서는 Riza를 이용해 Code Interpreter를 활용하는 방법을 설명합니다. 

[Riza - dashboard](https://dashboard.riza.io/)에 접속해서 credential을 발급 받습니다. Riza의 경우에 초기에는 무료로 이용할 수 있고, 트래픽이 늘어나면 유료로 활용 가능합니다. 이후 아래와 같은 패키지를 설치합니다. 

```text
pip install --upgrade --quiet langchain-community rizaio
```

Riza 환경에 추가로 패키지 필요한 경우에는 Custom Runtimes을 설정하여야 합니다. [Riza - dashboard](https://dashboard.riza.io/)에서 Custom Runtimes를 선택하여 아래와 같이 필요한 패키지를 설정합니다. 여기서는 pandas, numpy, matplotlib을 지정하였습니다. Runtime의 Revision ID는 코드에서 활용합니다. 

![image](https://github.com/user-attachments/assets/3763c905-bbf2-4e41-b86c-b0af6b818f5d)

Riza의 Credential과 Revision ID는 [cdk-agentic-workflow-stack.ts](./cdk-agentic-workflow/lib/cdk-agentic-workflow-stack.ts)와 같이 [secrets manager](https://aws.amazon.com/ko/secrets-manager/)에 등록하여 관리합니다. 

```java
const codeInterpreterSecret = new secretsmanager.Secret(this, `code-interpreter-secret-for-${projectName}`, {
  description: 'secret for code interpreter api key', // code interpreter
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  secretName: `code-interpreter-${projectName}`,
  secretObjectValue: {
    project_name: cdk.SecretValue.unsafePlainText(projectName),
    code_interpreter_api_key: cdk.SecretValue.unsafePlainText(''),
    code_interpreter_id: cdk.SecretValue.unsafePlainText(''),
  },
});
codeInterpreterSecret.grantRead(ec2Role) 
```

이후 [chat.py](./application/chat.py)와 같이 읽어서 RIZA_API_KEY로 등록해서 활용합니다.

```python
secretsmanager = boto3.client(
    service_name='secretsmanager',
    region_name=bedrock_region
)

code_interpreter_api_key = ""
get_code_interpreter_api_secret = secretsmanager.get_secret_value(
    SecretId=f"code-interpreter-{projectName}"
)
secret = json.loads(get_code_interpreter_api_secret['SecretString'])
code_interpreter_api_key = secret['code_interpreter_api_key']
code_interpreter_project = secret['project_name']
code_interpreter_id = secret['code_interpreter_id']

if code_interpreter_api_key:
    os.environ["RIZA_API_KEY"] = code_interpreter_api_key
```

Code Interpreter를 위해 code_interpreter와 code_drawer을 구현하였고, 아래와 같이 tools에 추가하여 활용합니다. code_interpreter는 python code를 실행한후 결과를 리턴하고, code_drawer는 matplotlib을 이용해 그래프를 만든 후에 Base64 이미지로 리턴합니다.

```python
tools = [code_drawer, code_interpreter]
```

Riza의 경우에 Code의 실행 결과가 stdout으로 전달되고 실행시 생성이 필요한 임시파일이나 이미지등을 루트에 저장할 수 없습니다. 따라서 아래와 같이 matplotlib을 위해 MPLCONFIGDIR을 /tmp로 설정하여야 합니다. 

```python
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'\n
```

또한 plt.savefig이 자동으로 생성되면 실행에 문제가 되므로 아래와 같이 제거합니다. 그래프를 파일로 저장하지 않더라도, buffer에 저장후 stdout으로 리턴할 수 있습니다.

code_drawer는 아래와 같이 구현할 수 있습니다. Riza가 제공하는 sandbox환경에서 외부 API 사용이 제한되므로 docstring엣 필요한 데이터는 code로 넣으라고 가이드하여야 합니다. 또한 matplotlib을 그림으로 저장시 한국어가 깨지는 문제점이 있으므로 아래와 같이 English를 사용하도록 가이드합니다. code_drawer에서는 얻어진 이미지를 S3에 저장한 후에 CloudFront의 도메인을 이용하여 URL 형태로 가공하고, 이후 streamlit에서 활용합니다. 


```python
from rizaio import Riza
@tool
def code_drawer(code):
    """
    Execute a Python script for draw a graph.
    Since Python runtime cannot use external APIs, necessary data must be included in the code.
    The graph should use English exclusively for all textual elements.
    When a comparison is made, all arrays must be of the same length.
    code: the Python code was written in English
    return: the url of graph
    """ 
        
    code = re.sub(r"seaborn", "classic", code)
    code = re.sub(r"plt.savefig", "#plt.savefig", code)
    
    pre = f"os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'\n"  # matplatlib
    post = """\n
import io
import base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.getvalue()).decode()

print(image_base64)
"""
    code = pre + code + post    
    logger.info(f"code: {code}")
    
    result = ""
    client = Riza()

    resp = client.command.exec(
        runtime_revision_id=chat.code_interpreter_id,
        language="python",
        code=code,
        env={
            "DEBUG": "true",
        }
    )
    output = dict(resp)

    base64Img = resp.stdout
    byteImage = BytesIO(base64.b64decode(base64Img))

    image_name = generate_short_uuid()+'.png'
    url = chat.upload_to_s3(byteImage, image_name)

    file_name = url[url.rfind('/')+1:]

    global image_url
    image_url.append(path+'/'+s3_image_prefix+'/'+parse.quote(file_name))

    result = f"생성된 그래프의 URL: {image_url}"
    return result
```

복잡한 작업을 코드로 수행하는 code_interpreter는 아래와 같이 구현합니다. 마찬가지로 English를 기본으로 사용하고 필요한 데이터는 코드에 포함하도록 요청합니다. 

```python
@tool
def code_interpreter(code):
    """
    Execute a Python script to solve a complex question.    
    Since Python runtime cannot use external APIs, necessary data must be included in the code.
    The Python runtime does not have filesystem access, but does include the entire standard library.
    Make HTTP requests with the httpx or requests libraries.
    Read input from stdin and write output to stdout."        
    code: the Python code was written in English
    return: the stdout value
    """ 
        
    code = re.sub(r"seaborn", "classic", code)
    code = re.sub(r"plt.savefig", "#plt.savefig", code)
    
    pre = f"os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'\n"  # matplatlib
    code = pre + code
    logger.info(f"code: {code}")
    
    client = Riza()

    resp = client.command.exec(
        runtime_revision_id=code_interpreter_id,
        language="python",
        code=code,
        env={
            "DEBUG": "true",
        }
    )
    output = dict(resp)
    print(f"output: {output}") # includling exit_code, stdout, stderr

    if resp.exit_code > 0:
        logger.debug(f"non-zero exit code {resp.exit_code}")

    resp.stdout        
    result = f"프로그램 실행 결과: {resp.stdout}"

    return result
```

### 실행 결과 

LLM에 "strawberry에 R은 몇개야?"로 질문하면 tokenizer의 특징으로 R은 2개라고 잘못된 답변을 합니다. Code Interpreter를 사용하면 코드를 통해서 R이 3개라고 정확한 답변을 할 수 있습니다. 메뉴에서 "Agent (Tool Use)"를 선택하고 아래와 같이 질문합니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/5e5b6e8e-bbca-4401-b971-65a733c51f2f" />

Agent의 정확한 동작을 LangSmith 로그를 이용해 확인합니다. Agent는 아래와 같은 code를 생성하여 code_interpreter를 실행시켰고, 결과적으로 정답인 3을 얻을 수 있었습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/f8f79e0a-710d-4db2-b201-03ee393f10d6" />



메뉴에서 "Agent"를 선택하고 "2000년 이후에 한국의 GDP 변화를 일본과 비교하는 그래프를 그려주세요."라고 입력하고 결과를 확인합니다. 이때 agent는 인터넷을 검색하여 얻어온 GDP정보를 code interpreter를 이용해 그래프로 표시합니다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/e2807786-5cda-4e4c-a645-252c397af97b" />


메뉴에서 "Agent"를 선택하고 "네이버와 카카오의 일별 주식 가격 변화량을 그래프로 비교해 주세요. 향후 투자 방법에 대한 가이드도 부탁드립니다."라고 입력후 결과를 확인하면 아래와 같습니다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/517237e4-3c7b-4649-b84e-940bc1fd34dd" />

## 활용 방법

EC2는 Private Subnet에 있으므로 SSL로 접속할 수 없습니다. 따라서, [Console-EC2](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)에 접속하여 "app-for-llm-streamlit"를 선택한 후에 Connect에서 sesseion manager를 선택하여 접속합니다. 

Github에서 app에 대한 코드를 업데이트 하였다면, session manager에 접속하여 아래 명령어로 업데이트 합니다. 

```text
sudo runuser -l ec2-user -c 'cd /home/ec2-user/agentic-workflow && git pull'
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
sudo runuser -l ec2-user -c "/home/ec2-user/.local/bin/streamlit run /home/ec2-user/agentic-workflow/application/app.py"
```



## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 

## 실행 결과

메뉴에서는 아래와 항목들을 제공하고 있습니다.

<img width="219" alt="image" src="https://github.com/user-attachments/assets/a016fb98-758a-4e27-9231-c207cebf3d1c" />


### RAG (Knowledge Base)

메뉴에서 RAG를 선택하고, "AWS의 스토리지 서비스에 대해 설명해주세요."라고 입력 후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/89dbe5f3-0dd1-4829-af80-a5fc51ad03e7)

이번에는 "Bedrock Agent와 S3를 비교해 주세요" 라고 입력후에 결과를 확인합니다. RAG만 적용한 경우에는 사용자의 질문을 그대로 검색하는데, 정확히 관련된 문서가 없으면 적절히 답변할 수 없습니다. 

![image](https://github.com/user-attachments/assets/a365357a-aaec-4745-ab74-fc3bcb769873)

메뉴에서 RAG를 선택하고 "Amazon Nova Pro 모델에 대해 설명해주세요"라고 입력하고 결과를 확인하면 아래와 같습니다. RAG는 retrieve - grade - generate의 단계를 통해 질문에 대한 답변 및 관련 문서를 제공합니다. RAG의 경우에 query decomposition을 하지 못하므로 입력된 질문을 Knowledge base로 구현된 RAG에 직접 질문하게 됩니다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/f84e4fa2-c7b8-492f-8c0c-82b32cb1426b" />


### 이미지 분석 

메뉴에서 [이미지 분석]과 모델로 [Claude 3.5 Sonnet]을 선택한 후에 [기다리는 사람들 사진](./contents/waiting_people.jpg)을 다운받아서 업로드합니다. 이후 "사진속에 있는 사람들은 모두 몇명인가요?"라고 입력후 결과를 확인하면 아래와 같습니다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/3e1ea017-4e46-4340-87c6-4ebf019dae4f" />


### Agentic Workflow

Agentic Workflow(Tool Use) 메뉴를 선택하여 오늘 서울의 날씨에 대해 질문을 하면, 아래와 같이 입력하고 결과를 확인합니다. LangGraph로 구현된 Tool Use 패턴의 agent는 날씨에 대한 요청이 올 경우에 openweathermap의 API를 요청해 날씨정보를 조회하여 활용할 수 있습니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/4693c1ff-b7e9-43f5-b7b7-af354b572f07" />

아래와 같은 질문은 LLM이 가지고 있지 않은 정보이므로, 인터넷 검색을 수행하고 그 결과로 아래와 같은 답변을 얻었습니다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/8f8d2e94-8be1-4b75-8795-4db9a8fa340f" />


RAG를 테스트 하였을때에 사용한 "Bedrock Agent와 S3를 비교해 주세요."라고 질문을 하면, 이번에는 좀더 나은 답변을 얻었습니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/969e9b84-5b80-4948-8627-f86bd2af26bc" />


메뉴에서 multi-agent collaboration을 선택한 후에 "지방 조직이 분비하는 exosome들이 어떻게 면역체계에 역할을 하고 어떻게 하면 좋은 exosome들을 분비시켜 당뇨나 병을 예방할수 있는지 알려주세요."라고 입력합니다. 이때 생성된 결과는 아래와 같습니다.

[지방 조직이 분비하는 exosome들이 어떻게 면역체계에 역할을 하고 어떻게 하면 좋은 exosome들을 분비시켜 당뇨나 병을 예방할수 있는지 알려주세요.](https://github.com/kyopark2014/agentic-workflow/blob/main/contents/%EC%A7%80%EB%B0%A9_%EC%A1%B0%EC%A7%81_exosome%EC%9D%98_%EB%A9%B4%EC%97%AD_%EC%97%AD%ED%95%A0_%EB%B0%8F_%EC%98%88%EB%B0%A9_%EB%B0%A9%EB%B2%95.md)


[(V2) 지방 조직이 분비하는 exosome들이 어떻게 면역체계에 역할을 하고 어떻게 하면 좋은 exosome들을 분비시켜 당뇨나 병을 예방할수 있는지 알려주세요.](./contents/지방_조직_exosome의_면역_역할과_예방_방법.md)

[남해여행](./contents/남해_여행_주제_또는_계획.md)

[세계 일주 방법](./contents/세계_일주_여행_방법_알아보기.md)


### Reference 

[Nova Pro User Guide](https://docs.aws.amazon.com/pdfs/nova/latest/userguide/nova-ug.pdf)

[Top Agentic AI Design Patterns](https://medium.com/@yugank.aman/top-agentic-ai-design-patterns-for-architecting-ai-systems-397798b44d5c)

[Explore Agent Recipes](https://www.agentrecipes.com/)

[AI Agent workflows from Anthropic](https://www.linkedin.com/posts/rakeshgohel01_these-ai-agent-workflows-from-anthropic-can-activity-7294724354221776897-pILx/?utm_source=share&utm_medium=member_android)

[LangGraph Multi-Agent Supervisor](https://github.com/langchain-ai/langgraph-supervisor)

[Multi-agent swarms with LangGraph](https://www.youtube.com/watch?v=JeyDrn1dSUQ)
