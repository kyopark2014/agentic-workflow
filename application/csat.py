import knowledge_base as kb
import utils
import operator
import chat
import traceback
import json
import re

from typing_extensions import Annotated, TypedDict
from typing import List, Tuple,Literal
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from multiprocessing import Process, Pipe
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

logger = utils.CreateLogger("chat")

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
    logger.info(f"total_score: {total_score}")
    st.info(f'주어진 문제는 모두 {len(json_data)}개의 절을 가지고 있고, 각 절의 점수 분포는 {scores}이며, 전체 점수는 {total_score}점 입니다.')
                            
    if chat.multi_region=="Enable":
        msg, earn_score = solve_problems_using_parallel_processing(json_data, st)

        logger.info(f"score: {earn_score}")
        msg += f"\n점수: {earn_score}점 / {total_score}점\n"

    else:
        total_idx = len(json_data)+1
        earn_score = total_available_score = 0
        #for idx, question_group in enumerate(json_data[:2]):
        for idx, question_group in enumerate(json_data):
            paragraph = question_group["paragraph"]
            logger.info(f"paragraph: {paragraph}")
            
            problems = question_group["problems"]
            logger.info(f"problems: {json.dumps(problems)}")
            
            result = solve_problems_in_paragraph(paragraph, problems, idx, total_idx, st)
            logger.info(f"result: {result}")

            idx = result["idx"]
            message = result["message"]
            score = result["score"]
            available_score = result["available_score"]
            logger.info(f"idx: {idx}")
            logger.info(f"message: {message}")
            logger.info(f"score: {score}")
            logger.info(f"available_score: {available_score}")
            
            msg += message
            earn_score += score
            total_available_score += available_score
            
            msg += "\n\n"
        
            st.warning(f"{idx+1}절까지 수행한 결과는 {earn_score} / {total_available_score}점입니다.")
        
        logger.info(f"score: {earn_score}")
        msg += f"\n점수: {earn_score}점 / {total_available_score}점\n"
    
    st.info(f"{msg}")

    return msg
    
def solve_problems_in_paragraph(paragraph, problems, idx, total_idx, st):
    message = f"{idx+1}/{total_idx}\n"
    
    earn_score = 0
    available_score = 0
    for n, problem in enumerate(problems):
        logger.info(f"--> problem[{n}]: {problem}")
    
        question = problem["question"]
        logger.info(f"question: {question}")
        question_plus = ""
        if "question_plus" in problem:
            question_plus = problem["question_plus"]
            logger.info(f"question_plus: {question_plus}")
        choices = problem["choices"]
        logger.info(f"choices: {choices}")
        correct_answer = problem["answer"]
        logger.info(f"correct_answer: {correct_answer}")
        score = problem["score"]
        logger.info(f"score: {score}")
        available_score += score

        selected_answer = solve_CSAT_Korean(paragraph, question, question_plus, choices, idx, n, correct_answer, score, st)
        logger.info(f"selected_answer: {selected_answer}")
                
        logger.info(f"pcorrect_answer: {correct_answer}, selected_answer: {selected_answer}")
        if correct_answer == selected_answer:
            message += f"{question} {selected_answer} (OK)\n"
            earn_score += int(score)
        else:
            message += f"{question} {selected_answer} (NOK, {correct_answer}, -{score})\n"
                    
    logger.info(f"earn_score: {earn_score}")
    logger.info(f"message: {message}")

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
    logger.info(f"total_idx: {total_idx}")
    
    messages = []
    earn_score = 0
    for idx in range(total_idx):
        messages.append("")
        
    for idx, question_group in enumerate(json_data[:1]):
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
        
        logger.info(f"idx:{idx} --> data:{question_group}")
        
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
        logger.info(f"result: {result}")
        
        idx = result["idx"]
        message = result["message"]
        score = result["score"]
        logger.info(f"idx:{idx} --> socre: {score}, message:{message}")

        if message is not None:
            logger.info(f"message: {message}")
            messages[idx] = message
            earn_score += score

    for process in processes:
        process.join()
            
    final_msg = ""   
    for message in messages:
        final_msg += message + '\n'
    
    logger.info(f"earn_score: {earn_score}")
    logger.info(f"final_msg: {final_msg}")
    
    return final_msg, earn_score

def solve_problems(conn, paragraph, problems, idx, total_idx, st):
    message = f"{idx+1}/{total_idx}\n"
    
    earn_score = available_score = 0    
    for n, problem in enumerate(problems):
        logger.info(f"--> problem[{n}]: {problem}")
    
        question = problem["question"]
        logger.info(f"question: {question}")
        question_plus = ""
        if "question_plus" in problem:
            question_plus = problem["question_plus"]
            logger.info(f"question_plus: {question_plus}")
        choices = problem["choices"]
        logger.info(f"choices: {choices}")
        correct_answer = problem["answer"]
        logger.info(f"correct_answer: {correct_answer}")
        score = problem["score"]
        logger.info(f"score: {score}")
        available_score += score

        selected_answer = solve_CSAT_Korean(paragraph, question, question_plus, choices, idx, n, correct_answer, score, st)
        logger.info(f"selected_answer: {selected_answer}")

        logger.info(f"correct_answer: {correct_answer}, selected_answer: {selected_answer}")
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
            
    logger.info(f"earn_score: {earn_score}")
    logger.info(f"message: {message}")

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
    com = re.compile('[0-9]') 
    value = com.findall(output)
    
    result = ""
    if not len(value) == 0:
        for v in value:
            result += v
        logger.info(f"result: {result}")
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
        repeat_counter: int

    def plan_node(state: State, config):
        logger.info(f"###### plan ######")
        logger.info(f"paragraph: {state['paragraph']}")
        logger.info(f"question: {state['question']}")
        logger.info(f"question_plus: {state['question_plus']}")

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
        logger.info(f"list_choices: {list_choices}")
        
        idx = config.get("configurable", {}).get("idx")
        nth = config.get("configurable", {}).get("nth")

        notification = f"({idx}-{nth}) 계획을 생성중입니다..."
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
        llm = chat.get_chat()
        planner = planner_prompt | llm
        response = planner.invoke({
            "paragraph": paragraph,
            "question": question,
            "question_plus": question_plus,
            "list_choices": list_choices
        })
        logger.info(f"esponse.content: {response.content}")
        result = response.content
        
        if not result.find('<plan>')==-1:
            output = result[result.find('<plan>')+6:result.find('</plan>')]
        else:
            output = result
        
        plan = output.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')
        logger.info(f"planning_steps: {planning_steps}")

        notification = f"({idx}-{nth}) 생성된 계획:\n\n {planning_steps}"
        logger.info(f"notification: {notification}")
        st.info(notification)
        
        return {
            "plan": planning_steps
        }

    def execute_node(state: State, config):
        logger.info(f"###### execute ######")
        plan = state["plan"]
        # print('plan: ', plan) 

        repeat_counter = state["repeat_counter"] if "repeat_counter" in state else 0        
        previous_answer = state["answer"] if "answer" in state else 0
        
        list_choices = ""
        choices = state["choices"]
        for i, choice in enumerate(choices):
            list_choices += f"({i+1}) {choice}\n"
        # print('list_choices: ', list_choices)    
                
        idx = config.get("configurable", {}).get("idx")
        nth = config.get("configurable", {}).get("nth")

        notification = f"({idx}-{nth}) 실행중인 계획: {plan[0]}"
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
            "당신은 국어 수능문제를 푸는 일타강사입니다."
            "매우 어려운 문제이므로 step by step으로 충분히 생각하고 답변합니다."
        )

        logger.info(f"model_type: {chat.model_type}")
        if chat.model_type=="claude":
            human = (
                "당신의 목표는 <paragraph> tag의 주어진 문장으로 부터 <question> tag의 주어진 질문에 대한 적절한 답변을 <choice> tag의 선택지에서 찾는것입니다."
                "답변은 반드시 <paragraph> tag의 주어진 문장, <question>의 주어진 질문, <choices>의 선택지를 참조하여 답변하고, <previous_result>의 이전 단계의 결과는 참고만 합니다. 절대 자신의 생각대로 답변하지 않습니다."                
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
        for attempt in range(3):
            try:
                llm = chat.get_chat()
                chain = prompt | llm                        
                response = chain.invoke({
                    "paragraph": state["paragraph"],
                    "question": state["question"],
                    "question_plus": state["question_plus"],
                    "list_choices": list_choices,
                    "info": context,
                    "task": task
                })
                logger.info(f"attempt: {attempt}, response.content: {response.content}")

                idx = config.get("configurable", {}).get("idx")
                nth = config.get("configurable", {}).get("nth")

                notification = f"({idx}-{nth}) 실행된 결과입니다.\n{response.content}"
                logger.info(f"notification: {notification}")
                st.info(notification)   
            
                result = response.content
                if not result.find('<confidence>')==-1:
                    output = result[result.find('<confidence>')+12:result.find('</confidence>')]
                    logger.info(f"output: {output}")
                    confidence = string_to_int(output)
                    logger.info(f"confidence: {confidence}")
                if not result.find('<result>')==-1:
                    output = result[result.find('<result>')+8:result.find('</result>')]
                    logger.info(f"output: {output}")
                    choice = string_to_int(output)
                    logger.info(f"choice: {choice}")
                break
            except Exception:
                response = AIMessage(content="답변을 찾지 못하였습니다.")

                err_msg = traceback.format_exc()
                logger.info(f"error message: {err_msg}")
        
        transaction = [HumanMessage(content=task), AIMessage(content=result)]
        # print('transaction: ', transaction)

        answer = choice if choice>0 and choice<6 else 0
        # print(f"previous_answer: {previous_answer}, answer: {answer}")        
        if previous_answer == answer: 
            repeat_counter += 1
            logger.info(f"repeat_counter: {repeat_counter}")   
    
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
        
        notification = f"({idx}-{nth}) 새로운 계획을 생성합니다..."
        logger.info(f"notification: {notification}")
        st.info(notification)
        
        system = (
            "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
        )        

        if chat.model_type=="claude":
            human = (
                "<paragraph> tag의 주어진 문장과 <question> tag의 주어진 질문 참조하여 <choices> tag의 선택지에서 거장 적절한 항목을 선택하기 위해서는 잘 세워진 계획이 있어야 합니다."
                "답변은 반드시 <paragraph> tag의 주어진 문장, <question>의 주어진 질문, <choices>의 선택지를 참조하여 답변하고, <previous_result>의 이전 단계의 결과는 참고만 합니다. 절대 자신의 생각대로 답변하지 않습니다."
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
        llm = chat.get_chat()
        replanner = replanner_prompt | llm        

        # plans = '\n'.join(state["plan"])
        # print('plans: ', plans)
        # past_steps = '\n'.join(state["past_steps"])
        # print('past_steps: ', past_steps)
        list_choices = ""
        choices = state["choices"]
        for i, choice in enumerate(choices):
            list_choices += f"({i+1}) {choice}\n\n"
        logger.info(f"prepalist_choicesre: {list_choices}")

        response = replanner.invoke({
            "paragraph": state["paragraph"],
            "question_plus": state["question_plus"],
            "question": state["question"],
            "list_choices": list_choices,
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

            notification = f"({idx}-{nth}) 생성된 계획:\n\n {planning_steps}"
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
        
        idx = config.get("configurable", {}).get("idx")
        nth = config.get("configurable", {}).get("nth")
        correct_answer = config.get("configurable", {}).get("correct_answer")
        score = config.get("configurable", {}).get("score")
        
        notification = f"({idx}-{nth}) 최종 답변을 구합니다..."
        logger.info(f"notification: {notification}")
        st.info(notification)
                
        answer = state["answer"]

        logger.info(f"answer: {answer}, correct_answer: {correct_answer}")
        logger.info(f"Type--> answer: {answer.__class__}, correct_answer: {correct_answer.__class__}")
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
        logger.info(f"context: {context}")
                                
        logger.info(f"paragraph: {state['paragraph']}")
        logger.info(f"prequestionpare: {state['question']}")
        logger.info(f"question_plus: {state['question_plus']}")
        
        list_choices = ""
        choices = state["choices"]
        for i, choice in enumerate(choices):
            list_choices += f"({i+1}) {choice}\n"
        logger.info(f"list_choices: {list_choices}")
        
        system = (
            "당신은 국어 수능문제를 푸는 일타강사입니다."
            "매우 어려운 문제이므로 step by step으로 충분히 생각하고 답변합니다."
        )    

        if chat.model_type=="claude":    
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
                llm = chat.get_chat()
                chain = prompt | llm                
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
                logger.info(f"attempt: {attempt}, result: {result}")

                notification = f"({idx}-{nth}) 최종으로 얻어진 결과:\n\n{result}"
                logger.info(f"notification: {notification}")
                st.info(notification)

                if not result.find('<confidence>')==-1:
                    output = result[result.find('<confidence>')+12:result.find('</confidence>')]
                    logger.info(f"output: {output}")

                    confidence = string_to_int(output)
                    logger.info(f"confidence: {confidence}")

                if not result.find('<result>')==-1:
                    output = result[result.find('<result>')+8:result.find('</result>')]
                    logger.info(f"output: {output}")
                    answer = string_to_int(output)
                    logger.info(f"answer: {answer}")
                break
            except Exception:
                    response = AIMessage(content="답변을 찾지 못하였습니다.")
                    err_msg = traceback.format_exc()
                    logger.info(f"error message: {err_msg}")

        logger.info(f"nswer: {answer}, correct_answer: {correct_answer}")
        logger.info(f"Type--> answer: {answer.__class__}, correct_answer: {correct_answer.__class__}")
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
            logger.info(f"Finished: {key}")
            #print("value: ", value)            
    logger.info(f"value: {value}")

    answer = value["answer"]
    logger.info(f"final answe: {answer}")
    
    notification = f"({idx}-{nth}) 최종 답변은 {answer}입니다."
    logger.info(f"notification: {notification}")
    st.info(notification)
        
    return answer
