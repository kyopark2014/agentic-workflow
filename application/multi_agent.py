import traceback
import boto3
import utils
import chat
import knowledge_base as kb
import search

from typing import List
from typing_extensions import TypedDict
from pydantic.v1 import BaseModel, Field
from multiprocessing import Process, Pipe
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph

logger = utils.CreateLogger("multi-agent")

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
        
        logger.info(f"###### reflect ######")
        draft = state['draft']
        
        idx = config.get("configurable", {}).get("idx")
        logger.info(f"reflect_node id: {idx}")

        if chat.debug_mode=="Enable":
            st.info(f"{idx}: draft에서 개선 사항을 도출합니다.")
    
        reflection = []
        search_queries = []
        for attempt in range(20):
            llm = chat.get_chat()
            if chat.isKorean(draft):
                structured_llm = llm.with_structured_output(ResearchKor, include_raw=True)
            else:
                structured_llm = llm.with_structured_output(Research, include_raw=True)
            
            try:
                # print('draft: ', draft)
                info = structured_llm.invoke(draft)
                logger.info(f"attempt: {attempt}, info: {info}")
                    
                if not info['parsed'] == None:
                    parsed_info = info['parsed']
                    # print('reflection: ', parsed_info.reflection)                
                    reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                    search_queries = parsed_info.search_queries

                    if chat.debug_mode=="Enable":
                        st.info(f"{idx}: 개선사항: {reflection}")
                    
                    logger.info(f"reflection: {parsed_info.reflection}")
                    logger.info(f"search_queries: {search_queries}")
            
                    if chat.isKorean(draft):
                        translated_search = []
                        for q in search_queries:
                            llm = chat.get_chat()
                            if chat.isKorean(q):
                                search = chat.traslation(llm, q, "Korean", "English")
                            else:
                                search = chat.traslation(llm, q, "English", "Korean")
                            translated_search.append(search)
                            
                        logger.info(f"translated_search: {translated_search}")
                        search_queries += translated_search

                    if chat.debug_mode=="Enable":
                        st.info(f"검색어: {search_queries}")

                    logger.info(f"search_queries (mixed): {search_queries}")
                    break
            except Exception:
                logger.info(f"--> parsing error")

                err_msg = traceback.format_exc()
                logger.info(f"error message: {err_msg}")
                # raise Exception ("Not able to request to LLM")
            
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "reflection": reflection,
            "search_queries": search_queries,
            "revision_number": revision_number + 1
        }
    
    def reflect_node2(state: ReflectionState, config):        
        logger.info(f"###### reflect ######")
        draft = state['draft']
        
        idx = config.get("configurable", {}).get("idx")
        logger.info(f"reflect_node idx:: {idx}")

        if chat.debug_mode=="Enable":
            st.info(f"{idx}: draft에서 개선 사항을 도출합니다.")
    
        reflection, search_queries = extract_reflection2(draft)
        if chat.debug_mode=="Enable":  
            st.info(f'개선할 사항: {reflection}')
            st.info(f'추가 검색어: {search_queries}')    

        if chat.isKorean(draft):
            translated_search = []
            for q in search_queries:
                llm = chat.get_chat()
                if chat.isKorean(q):
                    search = chat.traslation(llm, q, "Korean", "English")
                else:
                    search = chat.traslation(llm, q, "English", "Korean")
                translated_search.append(search)
                
            logger.info(f"translated_searc: {translated_search}")
            search_queries += translated_search

        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "reflection": reflection,
            "search_queries": search_queries,
            "revision_number": revision_number + 1
        }

    def retrieve_for_writing(conn, q, config):
        idx = config.get("configurable", {}).get("idx") 
         
        if chat.debug_mode=="Enable":
            st.info(f"검색을 수행합니다. 검색어: {q}")

        relevant_docs = kb.retrieve_documents_from_knowledge_base(q, top_k=chat.numberOfDocs)
        relevant_docs += search.retrieve_documents_from_tavily(q, top_k=chat.numberOfDocs)

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
        filtered_docs = chat.grade_documents(q, relevant_docs) # grading    
        filtered_docs = chat.check_duplication(filtered_docs) # check duplication
                                
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
        logger.info(f"parallel_retrieval: {parallel_retrieval}")
        
        if parallel_retrieval == 'enable':
            docs = parallel_retriever(search_queries, config)
        else:
            for q in search_queries:      
                if chat.debug_mode=="Enable":
                    st.info(f"검색을 수행합니다. 검색어: {q}")

                relevant_docs = kb.retrieve_documents_from_knowledge_base(q, top_k=chat.numberOfDocs)
                relevant_docs += search.retrieve_documents_from_tavily(q, top_k=chat.numberOfDocs)
            
                # grade
                docs += chat.grade_documents(q, relevant_docs) # grading
                    
            docs = chat.check_duplication(docs) # check duplication
            for i, doc in enumerate(docs):
                logger.info(f"#### {i}: {doc.page_content[:100]}")
                            
        return docs
        
    def get_revise_prompt(draft):
        if chat.isKorean(draft):
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

        llm = chat.get_chat()
        reflect_chain = revise_prompt | llm

        return reflect_chain
        
    def revise_draft(state: ReflectionState, config):   
        logger.info(f"###### revise_draft ######")
        
        draft = state['draft']
        search_queries = state['search_queries']
        reflection = state['reflection']
        logger.info(f"draft: {draft}")
        logger.info(f"search_queries: {search_queries}")
        logger.info(f"reflection: {reflection}")

        idx = config.get("configurable", {}).get("idx")
        logger.info(f"revise_draft idx: {idx}")
        
        if chat.debug_mode=="Enable":
            st.info(f"{idx}: 개선사항을 반영하여 새로운 답변을 생성합니다.")
        
        # reference = state['reference'] if 'reference' in state else []     
        if 'reference' in state:
            reference = state['reference'] 
        else:
            reference = []            

        if len(search_queries) and len(reflection):
            docs = retrieve_docs(search_queries, config)        
            logger.info(f"docs: {docs}")
                    
            content = []   
            if len(docs):                
                for d in docs:
                    content.append(d.page_content)            
                logger.info(f"content: {content}")
                                    
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
                logger.info(f"--> reflection: {reflection}")
                logger.info(f"--> revised_draft: {revised_draft}")

                st.info(f"revised_draft: {revised_draft}")

                reference += docs
                logger.info(f"len(reference): {len(reference)}")
            else:
                logger.info(f"No relevant document!")
                revised_draft = draft
        else:
            logger.info(f"No reflection!")
            revised_draft = draft
            
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        
        return {
            "revised_draft": revised_draft,            
            "revision_number": revision_number,
            "reference": reference
        }
        
    MAX_REVISIONS = 1
    def should_continue(state: ReflectionState, config):
        logger.info(f"###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        logger.info(f"max_revisions: {max_revisions}")
            
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
        logger.info(f"###### plan ######")
        instruction = state["instruction"]
        logger.info(f"subject: {instruction}")

        if chat.debug_mode=="Enable":
            st.info(f"계획을 생성합니다. 요청사항: {instruction}")
        
        if chat.isKorean(instruction):
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
                
        llm = chat.get_chat()
        
        planner = planner_prompt | llm
    
        response = planner.invoke({"instruction": instruction})
        logger.info(f"response: {response.content}")
    
        plan = response.content.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')        
        logger.info(f"planning_steps: {planning_steps}")

        if chat.debug_mode=="Enable":
            st.info(f"생성된 계획: {planning_steps}")
            
        return {
            "instruction": instruction,
            "planning_steps": planning_steps
        }
        
    def execute_node(state: State, config):
        logger.info(f"###### execute_node ######")    
        
        instruction = state["instruction"]        
        if chat.isKorean(instruction):
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
            logger.info(f"plan is too long")
            # print(plan)
            return
        
        text = ""
        drafts = []
        for idx, step in enumerate(planning_steps):            
            # Invoke the write_chain
            llm = chat.get_chat()
            write_chain = write_prompt | llm     

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

            if chat.debug_mode=="Enable":
                st.info(f"수행단계: {step}")
            
            if output.find('<result>')==-1:
                draft = output
            else:
                draft = output[output.find('<result>')+8:output.find('</result>')]

            if chat.debug_mode=="Enable":
                st.info(f"생성결과: {draft}")
                                              
            logger.info(f"--> step: {step}")
            logger.info(f"--> draft: {draft}")
                
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
            logger.info(f"idx: {idx}")

        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
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
            
            logger.info(f"idx:{idx} --> draft:{draft}")
            
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
                logger.info(f"result: {result}")
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
        
        llm = chat.get_chat()
        chain = prompt | llm    
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
            logger.info(f"error message: {err_msg}")                    
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
            logger.info(f"reference {i}: doc")
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
            logger.info(f"name: {name}")
            
            excerpt = ""+doc.page_content

            excerpt = re.sub('"', '', excerpt)
            logger.info(f"length: {len(excerpt)}")
            
            if name in nameList:
                logger.info(f"duplicated")
            else:
                reference = reference + f"{cnt}. [{name}]({url})"
                nameList.append(name)
                cnt = cnt+1
                
        return reference
    
    def revise_answer(state: State, config):
        logger.info(f"###### revise ######")
        drafts = state["drafts"]        
        logger.info(f"drafts: {drafts}")
        
        parallel_revise = config.get("configurable", {}).get("parallel_revise", "enable")
        logger.info(f"parallel_revise: {parallel_revise}")

        if chat.debug_mode=="Enable":
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
        
        logger.info(f"len(references): {len(references)}")
        
        # markdown file
        markdown_key = 'markdown/'+f"{subject}.md"
        # print('markdown_key: ', markdown_key)
        
        final_doc = f"## {state['instruction']}\n\n"+final_doc

        if references:
            logger.info(f"references: {references}")

            markdown_reference = chat.get_references(references)
            
            logger.info(f"markdown_reference: {markdown_reference}")

            final_doc += markdown_reference
                
        s3_client = boto3.client(
            service_name='s3',
            region_name=chat.bedrock_region
        )  
        response = s3_client.put_object(
            Bucket=chat.s3_bucket,
            Key=markdown_key,
            ContentType='text/markdown',
            Body=final_doc.encode('utf-8')
        )
        # print('response: ', response)
        
        markdown_url = f"{chat.path}/{markdown_key}"
        logger.info(f"markdown_url: {markdown_url}")
        
        # html file
        html_key = 'markdown/'+f"{subject}.html"
            
        html_body = markdown_to_html(final_doc)
        logger.info(f"html_body: {html_body}")
        
        s3_client = boto3.client(
            service_name='s3',
            region_name=chat.bedrock_region
        )  
        response = s3_client.put_object(
            Bucket=chat.s3_bucket,
            Key=html_key,
            ContentType='text/html',
            Body=html_body
        )
        # print('response: ', response)
        
        html_url = f"{chat.path}/{html_key}"
        logger.info(f"html_url: {html_url}")

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
        "parallel_revise": chat.multi_region
    }
    
    output = app.invoke(inputs, config)
    logger.info(f"output: {output}")
    
    return output['final_doc'], reference_docs

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
            llm = chat.get_chat()
            chain = critique_prompt | llm
            result = chain.invoke({
                "draft": draft
            })
            logger.info(f"result: {result}")

            output = result.content

            if output.find('<result>') != -1:
                output = output[output.find('<result>')+8:output.find('</result>')]
            logger.info(f"output: {output}")

            reflection = output            
            break
                
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}") 

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
            llm = chat.get_chat()
            if chat.isKorean(draft):
                structured_llm_queries = llm.with_structured_output(QueriesKor, include_raw=True)
            else:
                structured_llm_queries = llm.with_structured_output(Queries, include_raw=True)

            retrieval_quries = queries_prompt | structured_llm_queries
            
            info = retrieval_quries.invoke({
                "draft": draft,
                "reflection": reflection
            })
            logger.info(f"attempt: {attempt}, info: {info}")
                
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                logger.info(f"parsed_info: {parsed_info}")
                search_queries = parsed_info.search_queries
                logger.info(f"search_queries: {search_queries}")
            break
                
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}") 

    return reflection, search_queries

