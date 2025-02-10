import operator
import utils
import chat 
import csat
import traceback

from typing_extensions import Annotated, TypedDict
from typing import List, Tuple,Literal
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from multiprocessing import Process, Pipe
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

logger = utils.CreateLogger("csat")

def run_self_planning(question, st):    
    class State(TypedDict):
        plan: list[str]
        past_steps: Annotated[List[Tuple], operator.add]
        info: Annotated[List[Tuple], operator.add]
        paragraph: str
        question: str
        question_plus: str
        choices: list[str]
        answer: int
        repeat_counter: int

    def plan_node(state: State, config):
        logger.info(f"###### plan ######")
        logger.info(f"question: {state['question']}")
        
        question = state["question"]

        notification = f"계획을 생성중입니다..."
        logger.info(f"notification: {notification}")
        st.info(notification)
                                
        system = (
            "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
            
            "문제를 충분히 이해하고, 문제 해결을 위한 계획을 다음 형식으로 4단계 이하의 계획을 세웁니다."                
            "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
            "1. [질문을 해결하기 위한 단계]"
            "2. [질문을 해결하기 위한 단계]"
            "..."                
        )
        
        if chat.model_type=="claude":
            human = (
                "<question> tag의 주어진 질문에 대한 적절한 답변을 찾기 위한 단계별 계획을 세우세요."
                "결과에 <plan> tag를 붙여주세요."
                
                "주어진 질문:"
                "<question>"
                "{question}"                                
                "</question>"
            )
        else:
            human = (
                "Question에 대한 적절한 답변을 찾기 위한 단계별 계획을 세우세요."
                "결과에 <plan> tag를 붙여주세요."
                
                "Question:"
                "{question}"                                                
            )
                            
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )
        llm = chat.get_chat()
        planner = planner_prompt | llm
        response = planner.invoke({
            "question": question
        })
        logger.info(f"response.content: {response.content}")
        result = response.content
        
        if not result.find('<plan>')==-1:
            output = result[result.find('<plan>')+6:result.find('</plan>')]
        else:
            output = result
        
        plan = output.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')
        logger.info(f"planning_steps: {planning_steps}")

        st.info(f"생성된 계획: {planning_steps}")
        
        return {
            "plan": planning_steps
        }

    def execute_node(state: State, config):
        logger.info(f"###### execute ######")
        plan = state["plan"]
        # print('plan: ', plan) 

        repeat_counter = state["repeat_counter"] if "repeat_counter" in state else 0        
        previous_answer = state["answer"] if "answer" in state else 0
                
        notification = f"실행중인 계획: {plan[0]}"
        logger.info(f"notification: {notification}")
        st.info(notification)        
        
        task = plan[0]
        logger.info(f"task: {task}")                   
        
        context = ""
        for info in state['info']:
            if isinstance(info, HumanMessage):
                context += info.content+"\n"
            else:
                context += info.content+"\n\n"
        # print('context: ', context)
                        
        system = (
            "당신은 수학 수능문제를 푸는 일타강사입니다."
            "매우 어려운 문제이므로 step by step으로 충분히 생각하고 답변합니다."
        )

        logger.info(f"model_type: {chat.model_type}")
        if chat.model_type=="claude":
            human = (
                "당신의 목표는 <question> tag의 주어진 질문에 대한 적절한 답변을 찾는것입니다."
                "답변은 반드시 <question>의 주어진 질문에 대해 답변하고, <previous_result>의 이전 단계의 결과는 참고만 합니다. 절대 자신의 생각대로 답변하지 않습니다."                
                "<previous_result> tag에 있는 이전 단계의 결과를 참조하여, <task> tag의 실행 단계를 수행하고 적절한 답변을 구합니다."
                "적절한 답변을 찾지 못했다면 다시 한번 읽어보고 충분히 생각하고 답변합니다."
                "풀이 방법의 근거를 주어진 문장에서 찾아 설명하세요."
                "최종 결과의 번호에 <result> tag를 붙여주세요."
                "최종 결과의 신뢰도를 1-5 사이의 숫자로 나타냅니다. 신뢰되는 <confidence> tag를 붙입니다."  
                                    
                "주어진 질문:"
                "<question>"
                "{question}"                    
                "</question>"
                                
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
                "당신의 목표는 Question에 대한 적절한 답변을 Question에서 찾는것입니다."
                "답변은 반드시 Question을 참조하여 답변하고, Past Results는 참고만 합니다. 절대 자신의 생각대로 답변하지 않습니다."                
                "Past Results를 참조하여, Task를 수행하고 적절한 답변을 구합니다."                
                "적절한 답변을 고를 수 없다면 다시 한번 읽어보고 가장 가까운 것을 선택합니다." 
                "풀이 방법의 근거를 주어진 문장에서 찾아 설명하세요."
                "최종 결과의 번호에 <result> tag를 붙여주세요."
                "최종 결과의 신뢰도를 1-5 사이의 숫자로 나타냅니다. 신뢰되는 <confidence> tag를 붙입니다."  
                                    
                "Question:"
                "{question}"
                                
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

        confidence = 0
        answer = ""
        result = ""
        for attempt in range(3):
            try:
                llm = chat.get_chat()
                chain = prompt | llm                        
                response = chain.invoke({
                    "question": state["question"],
                    "info": context,
                    "task": task
                })
                logger.info(f"attempt: {attempt}, response.content: {response.content}")

                notification = f"(실행된 결과입니다.\n{response.content}"
                logger.info(f"notification: {notification}")
                st.info(notification)   
            
                result = response.content
                if not result.find('<confidence>')==-1:  # confidence
                    output = result[result.find('<confidence>')+12:result.find('</confidence>')]
                    logger.info(f"output: {output}")
                    confidence = csat.string_to_int(output)
                    logger.info(f"confidence: {confidence}")
                if not result.find('<result>')==-1: # result
                    answer = result[result.find('<result>')+8:result.find('</result>')]
                    logger.info(f"answer: {answer}")
                    st.info(f"{answer}")
                break
            except Exception:
                response = AIMessage(content="답변을 찾지 못하였습니다.")

                err_msg = traceback.format_exc()
                logger.info(f"error message: {err_msg}")
        
        transaction = [HumanMessage(content=task), AIMessage(content=result)]
        # print('transaction: ', transaction)

        # print(f"previous_answer: {previous_answer}, answer: {answer}")        
        if previous_answer == answer: 
            repeat_counter += 1
            logger.info(f"repeat_counter: {repeat_counter}")   
    
        # if confidence >= 4 and answer:
        #     plan = []  
        # else:
        #     plan = state["plan"]
        plan = state["plan"]
        
        return {
            "plan": plan,
            "info": transaction,
            "past_steps": [task],
            "answer": answer,
            "repeat_counter": repeat_counter
        }

    def replan_node(state: State, config):
        logger.info(f"#### replan ####")
        # print('state of replan node: ", state)        
        logger.info(f"past_steps: {state['past_steps']}")
                
        idx = config.get("configurable", {}).get("idx")
        nth = config.get("configurable", {}).get("nth")
        
        if len(state["plan"])==0:
            return {"plan": []}
        
        repeat_counter = state["repeat_counter"] if "repeat_counter" in state else 0
        logger.info(f"repeat_counter: {repeat_counter}")
        if repeat_counter >= 3:
            st.info("결과가 3회 반복되므로, 현재 결과를 최종 결과를 리턴합니다.")
            return {"plan": []}
        
        notification = f"새로운 계획을 생성합니다..."
        logger.info(f"notification: {notification}")
        st.info(notification)
        
        system = (
            "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
        )        

        if chat.model_type=="claude":
            human = (
                "<question> tag의 주어진 질문을 해결하기 위해 잘 세워진 계획이 있어야 합니다."
                "답변은 반드시 <question>의 주어진 질문에 대해 답변하고, <previous_result>의 이전 단계의 결과는 참고만 합니다. 절대 자신의 생각대로 답변하지 않습니다."
                "<original_plan> tag의 원래 계획을 상황에 맞게 수정하세요."
                "<question> tag의 주어진 질문을 참조하여 선택지에서 거장 적절한 항목을 선택하기 위해서는 잘 세워진 계획이 있어야 합니다."
                "<original_plan> tag에 있는 당신의 원래 계획에서 아직 수행되지 않은 계획들을 수정된 계획에 포함하세요."
                "수정된 계획에는 <past_steps> tag의 완료한 단계는 포함하지 마세요."
                "새로운 계획에는 <plan> tag를 붙여주세요."

                "주어진 질문:"
                "<question>"
                "{question}"
                "</question>"
                                
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
                "Question에 대한 적절한 답변을 위해서는 잘 세워진 계획이 있어야 합니다."
                "답변은 반드시 Question에 대해 답변하고, Past Results는 참고만 합니다. 절대 자신의 생각대로 답변하지 않습니다."
                "Original Steps를 상황에 맞게 수정하여 새로운 계획을 세우세요."
                "Original Steps의 첫번째 단계는 이미 완료되었으니 절대 포함하지 않습니다."
                "Original Plan에서 아직 수행되지 않은 단계를 새로운 계획에 포함하세요."
                "Past Steps의 완료한 단계는 계획에 포함하지 마세요."
                "새로운 계획에는 <plan> tag를 붙여주세요."
                
                "Question:"
                "{question}"
                
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
        llm = chat.get_chat()
        replanner = replanner_prompt | llm        

        # plans = '\n'.join(state["plan"])
        # print('plans: ', plans)
        # past_steps = '\n'.join(state["past_steps"])
        # print('past_steps: ', past_steps)
        
        response = replanner.invoke({
            "question": state["question"],
            "plan": state["plan"],
            "past_steps": state["past_steps"]
        })
        logger.info(f"response.content: {response.content}")
        result = response.content        
        
        if result.find('<plan>') == -1:
            logger.info(f"result: {result}")
            st.info(result)
            
            return {"plan":[], "repeat_counter": repeat_counter}
        else:
            output = result[result.find('<plan>')+6:result.find('</plan>')]
            logger.info(f"plan output: {output}")
            
            plans = output.strip().replace('\n\n', '\n')
            planning_steps = plans.split('\n')
            logger.info(f"planning_steps: {planning_steps}")

            notification = f"생성된 계획:\n\n {planning_steps}"
            logger.info(f"notification: {notification}")
            st.info(notification)
            
            return {"plan": planning_steps, "repeat_counter": repeat_counter}
                
    def should_end(state: State) -> Literal["continue", "end"]:
        logger.info(f"#### should_end ####")
        # print('state: ', state)
        
        plan = state["plan"]
        logger.info(f"plan: {plan}")
        if len(plan)<=1:
            next = "end"
        else:
            next = "continue"
        logger.info(f"should_end response: {next}")
        
        return next
        
    def final_answer(state: State, config) -> str:
        logger.info(f"#### final_answer ####")
        
        notification = f"최종 답변을 구합니다..."
        logger.info(f"notification: {notification}")
        st.info(notification)
                
        answer = state["answer"]

        if len(state["plan"])==0:
            return {"answer": answer}
        
        # get final answer
        info = state['info']
        
        context = ""
        for info in state['info']:
            if isinstance(info, HumanMessage):
                context += info.content+"\n"
            else:
                context += info.content+"\n\n"
        logger.info(f"context: {context}")
                                
        logger.info(f"question: {state['question']}")
        
        system = (
            "당신은 수학 수능문제를 푸는 일타강사입니다."
            "매우 어려운 문제이므로 step by step으로 충분히 생각하고 답변합니다."
        )    

        if chat.model_type=="claude":    
            human = (
                "<context> tag에 있는 검토 결과를 활용하여, <question> tag의 주어진 질문에 대한 적절한 답변을하고자 합니다."
                "답변을 모른다면 다시 한번 읽어보고 충분히 생각한 후에 답변합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "최종 결과의 번호에 <result> tag를 붙여주세요."
                "최종 결과의 신뢰도를 1-5 사이의 숫자로 나타냅니다. 신뢰되는 <confidence> tag를 붙입니다."  
                
                "이전 단계에서 검토한 결과:"
                "<context>"
                "{context}"
                "</context>"
                                
                "주어진 질문:"
                "<question>"
                "{question}"
                "</question>"
            )
        else:
            human = (
                "당신은 Past Results를 활용하여, Question에 대한 적절한 답변을 하고자 합니다."
                "답변을 모른다면 다시 한번 읽어보고 충분히 생각한 후에 답변합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "최종 결과 번호에 <result> tag를 붙여주세요. 예) <result>1</result>"  
                "최종 결과의 신뢰도를 1-5 사이의 숫자로 나타냅니다. 신뢰되는 <confidence> tag를 붙입니다."  
                
                "Past Results:"
                "{context}"
                                
                "Question:"
                "{question}"                
            )
                
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

        answer = 0
        for attempt in range(3):
            try:
                llm = chat.get_chat()
                chain = prompt | llm                
                response = chain.invoke(
                    {
                        "context": context,
                        "question": state["question"],
                    }
                )
                result = response.content
                logger.info(f"attempt: {attempt}, result: {result}")

                notification = f"최종으로 얻어진 결과:\n\n{result}"
                logger.info(f"notification: {notification}")
                st.info(notification)

                if not result.find('<confidence>')==-1:
                    output = result[result.find('<confidence>')+12:result.find('</confidence>')]
                    logger.info(f"output: {output}")

                    confidence = csat.string_to_int(output)
                    logger.info(f"confidence: {confidence}")

                if not result.find('<result>')==-1:
                    output = result[result.find('<result>')+8:result.find('</result>')]
                    logger.info(f"output: {output}")
                    answer = csat.string_to_int(output)
                    logger.info(f"answer: {answer}")
                break
            except Exception:
                    response = AIMessage(content="답변을 찾지 못하였습니다.")
                    err_msg = traceback.format_exc()
                    logger.info(f"error message: {err_msg}")

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
        "question": question
    }
    config = {
        "recursion_limit": 50
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            logger.info(f"Finished: {key}")
            #print("value: ", value)            
    logger.info(f"value: {value}")

    answer = value["answer"]
    logger.info(f"final answe: {answer}")
    
    notification = f"최종 답변은 {answer}입니다."
    logger.info(f"notification: {notification}")
    st.info(notification)
        
    return answer
