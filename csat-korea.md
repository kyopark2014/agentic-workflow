# 모델명 한국어 능력 평가

## 복잡한 문제로 수능 국어를 선택한 이유

대학 수학 능력 시험의 국어 영역은 LLM 모델의 한국어 능력을 측정하기에 좋은 주제입니다. 여기서는 [대학수학능력시험-국어영역](https://github.com/NomaDamas/KICE_slayer_AI_Korean/blob/master/data/2023_11_KICE.json)에 있는 문제와 답을 planning 패턴을 사용하는 agent를 이용해 풀어보고자 합니다. [2023년 수능의 국어(화법과 작문)의 1등급 컷](https://www.nextplay.kr/news/articleView.html?idxno=4617)은 92점이었습니다. 결과적으로 Claude 3.5 Sonnet(v2), Nova Pro 순으로 좋은 결과를 얻었습니다. Nova Pro의 결과는 Claude 3.5 Sonnet (v1)보다 약간 좋았고, Claude 3.5 Sonnet보다도 좋은 결과를 보였습니다.

여기서 사용한 planning 패턴의 동작은 아래 activity diagram을 참조합니다.

![image](https://github.com/user-attachments/assets/e81d9a77-1dc4-490a-a8a6-491eea5c15e0)

상세한 코드는 [chat.py](./application/chat.py)의 solve_CSAT_problem() 함수를 참조합니다. 여기서 workflow는 아래와 같이 정의하였습니다.

```python
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
```

이를 실행할 때에는 아래와 같이 합니다. 문제에 대한 정보인 question, question_plus, paragraph, choices은 업로드한 json파일에서 추출하여 활용합니다.

```python
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

answer = value["answer"]
```

여기서 plan 노드는 아래와 같이 정의하였습니다. Plan 노드를 통해 얻어진 계획은 CoT의 형태로 결과를 향상시킵니다.

```python
def plan_node(state: State, config):
    list_choices = ""
    choices = state["choices"]
    for i, choice in enumerate(choices):
        list_choices += f"({i+1}) {choice}\n"
    
    system = (
        "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
        
        "문제를 충분히 이해하고, 문제 해결을 위한 계획을 다음 형식으로 4단계 이하의 계획을 세웁니다."                
        "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
        "1. [질문을 해결하기 위한 단계]"
        "2. [질문을 해결하기 위한 단계]"
        "..."                
    )
    
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
    result = response.content
    if not result.find('<plan>')==-1:
        output = result[result.find('<plan>')+6:result.find('</plan>')]
    else:
        output = result    
    plan = output.strip().replace('\n\n', '\n')
    planning_steps = plan.split('\n')

    notification = f"({idx}-{nth}) 생성된 계획:\n\n {planning_steps}"
    st.info(notification)    
    return {
        "plan": planning_steps
    }
```

Execute 노드는 아래와 같이 구성합니다.이 노드에서는 plan 노드에서 생성한 계획중에 첫번째 계획을 수행합니다. 이때 신뢰도(confidence)를 평가하여 일정 기준 이상이면 다음 plan을 진행하지 않고 종료하고, 이하 일 경우에는 다음 plan을 순차적으로 수행합니다.

```python
def execute_node(state: State, config):
   plan = state["plan"]
   repeat_counter = state["repeat_counter"] if "repeat_counter" in state else 0        
   previous_answer = state["answer"] if "answer" in state else 0
   
   list_choices = ""
   choices = state["choices"]
   for i, choice in enumerate(choices):
       list_choices += f"({i+1}) {choice}\n"
           
   task = plan[0]
   context = ""
   for info in state['info']:
       if isinstance(info, HumanMessage):
           context += info.content+"\n"
       else:
           context += info.content+"\n\n"
                   
   system = (
       "당신은 국어 수능문제를 푸는 일타강사입니다."
       "매우 어려운 문제이므로 step by step으로 충분히 생각하고 답변합니다."
   )
   human = (
       "당신의 목표는 Paragraph으로 부터 Question에 대한 적절한 답변을 Question에서 찾는것입니다."
       "답변은 반드시 Paragraph, Question, List Choices을 참조하여 답변하고, Past Results는 참고만 합니다. 절대 자신의 생각대로 답변하지 않습니다."                
       "Past Results를 참조하여, Task를 수행하고 적절한 답변을 구합니다."                
       "적절한 답변을 고를 수 없다면 다시 한번 읽어보고 가장 가까운 것을 선택합니다." 
       "받드시 List Choices중에 하나를 선택하여 1-5 사이의 숫자로 답변합니다."
       "문제를 풀이할 때 모든 List Choices마다 근거를 주어진 문장에서 찾아 설명하세요."
       "List Choices의 선택지의 주요 단어들의 의미를 Paragraph와 비교해서 자세히 차이점을 찾습니다."
       "List Choices의 선택지를 모두 검토한 후에 최종 결과를 결정합니다."                
       "최종 결과의 번호에 <result> tag를 붙여주세요."
       "최종 결과의 신뢰도를 1-5 사이의 숫자로 나타냅니다. 신뢰되는 <confidence> tag를 붙입니다."  
                           
       "Paragraph:"
       "{paragraph}"

       "{question_plus}"
           
       "Question:"
       "{question}"
                       
       "List Choices:"
       "{list_choices}"
       
       "Past Results:"
       "{info}"

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
   result = response.content
   if not result.find('<confidence>')==-1:
       output = result[result.find('<confidence>')+12:result.find('</confidence>')]
       confidence = string_to_int(output)
   if not result.find('<result>')==-1:
       output = result[result.find('<result>')+8:result.find('</result>')]
       choice = string_to_int(output)
   transaction = [HumanMessage(content=task), AIMessage(content=result)]

   answer = choice if choice>0 and choice<6 else 0
   if previous_answer == answer: 
       repeat_counter += 1
       print("repeat_counter: ", repeat_counter)        

   if confidence >= 4:
       plan = []  
   else:
       plan = state["plan"]
   
   return {
       "plan": plan,
       "info": transaction,
       "past_steps": [task],
       "answer": answer,
       "repeat_counter": repeat_counter
   }
```

처음 새운 계획은 아래와 같이 replan 노드에서 지속적으로 개선합니다.

```python
def replan_node(state: State, config):
    if len(state["plan"])==0:
        return {"plan": []}
    
    repeat_counter = state["repeat_counter"] if "repeat_counter" in state else 0
    if repeat_counter >= 3:
        st.info("결과가 3회 반복되므로, 현재 결과를 최종 결과를 리턴합니다.")
        return {"plan": []}
    system = (
        "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
    )        
    human = (
        "Paragraph과 Question을 참조하여 List Choices에서 거장 적절한 항목을 선택하기 위해서는 잘 세워진 계획이 있어야 합니다."
        "답변은 반드시 Paragraph, Question, List Choices을 참조하여 답변하고, Past Results는 참고만 합니다. 절대 자신의 생각대로 답변하지 않습니다."
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

    list_choices = ""
    choices = state["choices"]
    for i, choice in enumerate(choices):
        list_choices += f"({i+1}) {choice}\n\n"

    response = replanner.invoke({
        "paragraph": state["paragraph"],
        "question_plus": state["question_plus"],
        "question": state["question"],
        "list_choices": list_choices,
        "plan": state["plan"],
        "past_steps": state["past_steps"]
    })
    result = response.content        
    
    if result.find('<plan>') == -1:
        print('result: ', result)
        st.info(result)        
        return {"plan":[], "repeat_counter": repeat_counter}
    else:
        output = result[result.find('<plan>')+6:result.find('</plan>')]        
        plans = output.strip().replace('\n\n', '\n')
        planning_steps = plans.split('\n')
        return {"plan": planning_steps, "repeat_counter": repeat_counter}
```


## 모델별로 수학능력시험 국어 영역 비교

### 시험 방법

아래와 같이 대화 형태로 RAG를 선택합니다.

![image](https://github.com/user-attachments/assets/99051bad-3532-4204-a234-42561a273067)

아래와 같이 1) 사용 모델을 지정하고 2) CSAT evaluator를 enable 합니다. 이후 [Browse files]를 선택하여 [국어 영역 문제](https://github.com/NomaDamas/KICE_slayer_AI_Korean/blob/master/data/2023_11_KICE.json)을 업로드합니다. 

![noname](https://github.com/user-attachments/assets/7ac16a88-b0cf-4ad6-9131-09b2dc6dca15)

이후 자동으로 수능 문제를 풀기 시작하는데, 같은 동작을 2-3회 반복하면서 결과를 모읍니다.

### 시험 결과
 
아래와 같이 가장 좋은 결과는 Claude 3.5 Sonnet(v2)이었고 2번째로 좋은 모델은 Nova Pro입니다. 이 모델은 Claude 3.5 Sonnet (v1)과 동급의 성능을 보여주고 있습니다. Claude 3.5 Haiku도 괜찮은 성적을 얻지만, Nova Lite나 Claude 3.0 Sonnet은 다른 모델대비 낮은 점수를 얻었습니다.

<img src="https://github.com/user-attachments/assets/571eb34a-f40e-4bec-b610-ee30ed73ab87" width="700">


이때의 정답, 오답의 내용은 아래와 같습니다.

#### Claude 3.5 Sonnet (v2)

<img src="https://github.com/user-attachments/assets/d767e581-a0bc-4752-b972-44aaec991ed1" width="600">

#### Nova Pro 

<img src="https://github.com/user-attachments/assets/b62df1ee-2b95-47da-b7fe-d2f7bc04c88d" width="600">

#### Claude 3.5 Haiku

<img src="https://github.com/user-attachments/assets/07b11094-3b16-484b-bb58-e768ebe2250a" width="600">

## Reference

[지문과 선택지-화법과 작문](https://github.com/NomaDamas/KICE_slayer_AI_Korean/blob/master/data/2023_11_KICE.json)

[수능 문제의 경우에 정답이 알려져있고 상세한 해설서](https://m.blog.naver.com/awesome-2030/222931282476)

[2023년 수능의 국어(화법과 작문)의 1등급 컷](https://www.nextplay.kr/news/articleView.html?idxno=4617)
