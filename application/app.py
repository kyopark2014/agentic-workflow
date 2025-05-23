import streamlit as st 
import chat
import knowledge_base as kb
import tool_use
import utils
import reflection
import planning
import deep_research
import csat
import cost_analysis as cost
import supervisor

# logging
logger = utils.CreateLogger("streamlit")

# title
st.set_page_config(page_title='Agentic Workflow', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "RAG": [
        "Bedrock Knowledge Base를 이용해 구현한 RAG로 필요한 정보를 검색합니다."
    ],
    "Agent (Tool Use)": [
        "Tool Use 방식의 Workflow를 수행하는 Agent를 구현합니다. 여기에서는 날씨, 시간, 도서추천, RAG, 인터넷 검색을 제공합니다."
    ],
    "Agent with Chat (Tool Use)": [
        "대화이력을 포함하여 Tool Use 방식의 Workflow를 수행하는 Agent를 구현합니다."
    ],
    "Agent (Reflection)": [
        "Reflection Workflow를 수행하는 Agent을 구현합니다."
    ],
    "Agent (Planning)": [
        "Planning Workflow를 수행하는 Agent을 구현합니다."
    ],
    "Deep Research Agent": [
        "Multi-agent Collaboration을 이용하여 Deep Research Agent를 구현합니다."
    ],
    "Supervisor": [
        "Supervisor 패턴의 multi-agent를 이용해 다양한 형태의 대화를 구현합니다."
    ],
    "번역하기": [
        "한국어와 영어에 대한 번역을 제공합니다. 한국어로 입력하면 영어로, 영어로 입력하면 한국어로 번역합니다."        
    ],
    "이미지 분석": [
        "이미지를 업로드하면 이미지의 내용을 요약할 수 있습니다."
    ],
    "이미지 문제 풀이": [
        "이미지를 업로드하면 planning agent를 이용해 문제를 풉니다."
    ],
    "비용 분석": [
        "Cloud 사용에 대한 분석을 수행합니다."
    ]
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Amazon Bedrock을 이용해 다양한 형태의 대화를 구현합니다." 
        "여기에서는 일상적인 대화와 각종 툴을 이용해 Agent를 구현할 수 있습니다." 
        "또한 번역이나 문법 확인과 같은 용도로 사용할 수 있습니다."
        "주요 코드는 LangChain과 LangGraph를 이용해 구현되었습니다.\n"
        "상세한 코드는 [Github](https://github.com/kyopark2014/langgraph-nova)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    mode = st.radio(
        label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "RAG", "Agent (Tool Use)", "Agent with Chat (Tool Use)", "Agent (Reflection)", "Agent (Planning)", "Deep Research Agent", "번역하기", "이미지 분석", "이미지 문제 풀이", "비용 분석"], index=0
        # label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "RAG", "Agent (Tool Use)", "Agent with Chat (Tool Use)", "Agent (Reflection)", "Agent (Planning)", "Deep Research Agent", "Supervisor", "번역하기", "이미지 분석", "이미지 문제 풀이", "비용 분석"], index=0
    )   
    st.info(mode_descriptions[mode][0])    
    # print('mode: ', mode)

    # model selection box
    if mode == '이미지 분석' or mode=="이미지 문제 풀이" or mode=="Agent (Tool Use)":
        index = 2
    elif mode == "Deep Research Agent":
        index = 6
    else:
        index = 2   
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ('Nova Pro', 'Nova Lite', 'Claude 3.7 Sonnet', 'Claude 3.5 Sonnet v1', 'Claude 3.5 Sonnet v2', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku'), index=index
    )
    uploaded_file = None
    if mode=='이미지 분석' or mode=="이미지 문제 풀이":
        st.subheader("🌇 이미지 업로드")
        uploaded_file = st.file_uploader("이미지 요약을 위한 파일을 선택합니다.", type=["png", "jpg", "jpeg"])
    elif mode=='RAG' or mode=="Agent (Tool Use)":
        st.subheader("📋 문서 업로드")
        # print('fileId: ', chat.fileId)
        uploaded_file = st.file_uploader("RAG를 위한 파일을 선택합니다.", type=["pdf", "txt", "py", "md", "csv", "json"], key=chat.fileId)

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    #print('debugMode: ', debugMode)

    # multi region check box
    select_multiRegion = st.checkbox('Multi Region', value=False)
    multiRegion = 'Enable' if select_multiRegion else 'Disable'
    #print('multiRegion: ', multiRegion)

    # internet checkbox
    select_internet = st.checkbox('Internet', value=True)
    internetMode = 'Enable' if select_internet else 'Disable'
    #print('internetMode: ', internetMode)

    # extended thinking of claude 3.7 sonnet
    select_reasoning = st.checkbox('Reasonking (only Claude 3.7 Sonnet)', value=False)
    reasoningMode = 'Enable' if select_reasoning and modelName=='Claude 3.7 Sonnet' else 'Disable'
    logger.info(f"reasoningMode: {reasoningMode}")

    # chart checkbox 
    selected_chart = st.checkbox('Chart', value=False)
    chart = 'Enable' if selected_chart else 'Disable'
    #print('chart: ', chart)

    chat.update(modelName, debugMode, multiRegion, internetMode, reasoningMode)
    
    # code interpreter checkbox
    select_csat_evaluator = st.checkbox('CSAT evaluator', value=False)
    CSAT_evaluator = 'Enable' if select_csat_evaluator else 'Disable'
    #print('CSAT_evaluator: ', CSAT_evaluator)

    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    # print('clear_button: ', clear_but

st.title('🔮 '+ mode)

if clear_button==True:
    chat.initiate()
    cost.cost_data = {}
    cost.visualizations = {}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages():
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "images" in message:                
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)
            st.markdown(message["content"])

display_chat_messages()

def show_references(reference_docs):
    if debugMode == "Enable" and reference_docs:
        with st.expander(f"답변에서 참조한 {len(reference_docs)}개의 문서입니다."):
            for i, doc in enumerate(reference_docs):
                st.markdown(f"**{doc.metadata['name']}**: {doc.page_content}")
                st.markdown("---")

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    uploaded_file = None
    
    st.session_state.greetings = False
    st.rerun()

    chat.clear_chat_history()

if chart == 'Enable':
    if mode == 'Agent (Tool Use)':
        col1, col2, col3 = st.columns([0.1, 0.25, 0.1])
        url = "https://raw.githubusercontent.com/kyopark2014/agentic-workflow/main/contents/tool_use.png"
        col2.image(url)
    elif mode == 'Agent (Reflection)':
        col1, col2, col3 = st.columns([0.2, 0.3, 0.2])
        url = "https://raw.githubusercontent.com/kyopark2014/agentic-workflow/main/contents/reflection.png"
        col2.image(url)    
    elif mode == 'Agent (Planning)':
        col1, col2, col3 = st.columns([0.2, 0.3, 0.2])
        url = "https://raw.githubusercontent.com/kyopark2014/agentic-workflow/main/contents/planning.png"
        col2.image(url)
    elif mode == 'Deep Research Agent':
        col1, col2, col3 = st.columns([0.1, 2.0, 0.1])    
        url = "https://raw.githubusercontent.com/kyopark2014/agentic-workflow/main/contents/multi_agent_collaboration.png"
        col2.image(url)
           
# Preview the uploaded image in the sidebar
file_name = ""
if uploaded_file and clear_button==False:
    logger.info(f"ploaded_file.name: {uploaded_file.name}")
    logger.info(f"SAT_evaluator: {CSAT_evaluator}")

    # upload and summary
    if uploaded_file.name and CSAT_evaluator=="Disable" and not (mode=='이미지 분석' or mode=="이미지 문제 풀이"):
        chat.initiate()
        if debugMode=='Enable':
            status = '선택한 파일을 업로드합니다.'
            logger.info(f"status: {status}")
            st.info(status)

        file_name = uploaded_file.name
        file_url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        logger.info(f"file_url: {file_url}")

        kb.sync_data_source()  # sync uploaded files
            
        status = f'선택한 "{file_name}"의 내용을 요약합니다.'
        # my_bar = st.sidebar.progress(0, text=status)
        
        # for percent_complete in range(100):
        #     time.sleep(0.2)
        #     my_bar.progress(percent_complete + 1, text=status)
        if debugMode=='Enable':
            logger.info(f"status: {status}")
            st.info(status)
    
        msg = chat.get_summary_of_uploaded_file(file_name, st)
        st.session_state.messages.append({"role": "assistant", "content": f"선택한 문서({file_name})를 요약하면 아래와 같습니다.\n\n{msg}"})    
        logger.info(f"msg: {msg}")
        st.rerun()
        
    if uploaded_file and clear_button==False and (mode=='이미지 분석' or mode=="이미지 문제 풀이"): # upload only
        st.image(uploaded_file, caption="이미지 미리보기", use_container_width=True)

        file_name = uploaded_file.name
        url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        logger.info(f"url: {url}")

    elif uploaded_file.name and CSAT_evaluator=="Enable" and uploaded_file.name.lower().endswith((".json")):   
        guide = "CSAT Evaluation을 시작합니다."
        st.write(guide)
        st.session_state.messages.append({"role": "assistant", "content": guide})
        state_of_CSAT_evaluator = True
        csat.solve_CSAT_problem(uploaded_file.getvalue(), st)

if clear_button==False and mode == '비용 분석':
    st.subheader("📈 Cost Analysis")

    if not cost.visualizations:
        cost.get_visualiation()

    if 'service_pie' in cost.visualizations:
        st.plotly_chart(cost.visualizations['service_pie'])
    if 'daily_trend' in cost.visualizations:
        st.plotly_chart(cost.visualizations['daily_trend'])
    if 'region_bar' in cost.visualizations:
        st.plotly_chart(cost.visualizations['region_bar'])

    with st.status("thinking...", expanded=True, state="running") as status:
        if not cost.cost_data:
            st.info("비용 데이터를 가져옵니다.")
            cost_data = cost.get_cost_analysis()
            logger.info(f"cost_data: {cost_data}")
            cost.cost_data = cost_data
        else:
            if not cost.insights:        
                st.info("잠시만 기다리세요. 지난 한달간의 사용량을 분석하고 있습니다...")
                insights = cost.generate_cost_insights()
                logger.info(f"insights: {insights}")
                cost.insights = insights
            
            st.markdown(cost.insights)
            st.session_state.messages.append({"role": "assistant", "content": cost.insights})

# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {prompt}")
    
    with st.chat_message("assistant"):
        if mode == '일상적인 대화':
            output = chat.general_conversation(prompt)            
            if reasoningMode=="Enable":
                with st.status("thinking...", expanded=True, state="running") as status:    
                    # extended thinking
                    if debugMode=="Enable":
                        chat.show_extended_thinking(st, output)

                    response = output.content
                    st.write(response)
                
            else:
                response = st.write_stream(output)
            logger.info(f"response: {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})
            # st.rerun()

            chat.save_chat_history(prompt, response)

        elif mode == 'RAG':
            with st.status("running...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_rag_with_knowledge_base(prompt, st)                           
                st.write(response)
                logger.info(f"response: {response}")

                st.session_state.messages.append({"role": "assistant", "content": response})

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Agent (Tool Use)':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, image_url, reference_docs = tool_use.run_agent_executor(prompt, "Disable", st)
                st.write(response)
                logger.info(f"response: {response}")

                if len(image_url):
                    for url in image_url:
                        logger.info(f"url: {url}")

                        file_name = url[url.rfind('/')+1:]
                        st.image(url, caption=file_name, use_container_width=True)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "images": image_url if image_url else []
                })

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 
        
        elif mode == 'Agent with Chat (Tool Use)':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, image_url, reference_docs = tool_use.run_agent_executor(prompt, "Enable", st)
                st.write(response)
                logger.info(f"response: {response}")

                if len(image_url):
                    for url in image_url:
                        logger.info(f"url: {url}")

                        file_name = url[url.rfind('/')+1:]
                        st.image(url, caption=file_name, use_container_width=True)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "images": image_url if image_url else []
                })

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 
        
        elif mode == 'Agent (Reflection)':
            with st.status("thinking...", expanded=True, state="running") as status:
                # esponse, reference_docs = chat.run_knowledge_guru(prompt, st)
                response, reference_docs = reflection.run_reflection(prompt, st)     
                st.write(response)
                logger.info(f"response: {response}")

                if response.find('<thinking>') != -1:
                    logger.info(f"Remove <thinking> tag.")
                    response = response[response.find('</thinking>')+12:]
                    logger.info(f"response without tag: {response}")

                st.session_state.messages.append({"role": "assistant", "content": response})

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Agent (Planning)':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, reference_docs = planning.run_planning(prompt, st)
                st.write(response)
                logger.info(f"response: {response}")

                if response.find('<thinking>') != -1:
                    logger.info(f"Remove <thinking> tag.")
                    response = response[response.find('</thinking>')+12:]
                    logger.info(f"response without tag: {response}")

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Deep Research Agent':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, reference_docs = deep_research.run_deep_research_agent(prompt, st)
                st.write(response)
                logger.info(f"response: {response}")

                if response.find('<thinking>') != -1:
                    logger.info(f"Remove <thinking> tag.")
                    response = response[response.find('</thinking>')+12:]
                    logger.info(f"response without tag: {response}")

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Supervisor':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, image_url, reference_docs = supervisor.run_supervisor(prompt, st)
                st.write(response)
                logger.info(f"response: {response}")

                if len(image_url):
                    for url in image_url:
                        logger.info(f"url: {url}")

                        file_name = url[url.rfind('/')+1:]
                        st.image(url, caption=file_name, use_container_width=True)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "images": image_url if image_url else []
                })

                # chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == '번역하기':
            response = chat.translate_text(prompt, modelName)
            st.write(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        elif mode == '이미지 분석':
            if uploaded_file is None or uploaded_file == "":
                st.error("파일을 먼저 업로드하세요.")
                st.stop()

            else:
                if modelName == "Claude 3.5 Haiku":
                    st.error("Claude 3.5 Haiku은 이미지를 지원하지 않습니다. 다른 모델을 선택해주세요.")
                else:
                    with st.status("thinking...", expanded=True, state="running") as status:
                        summary = chat.get_image_summarization(file_name, prompt, st)
                        st.write(summary)

                        st.session_state.messages.append({"role": "assistant", "content": summary})
                        # st.rerun()

        elif mode == "이미지 문제 풀이":
            if uploaded_file is None or uploaded_file == "":
                st.error("파일을 먼저 업로드하세요.")
                st.stop()

            else:
                with st.status("thinking...", expanded=True, state="running") as status:
                    answer = chat.solve_image_question(file_name, prompt, st)
                    st.write(answer)

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    # st.rerun()

        elif mode == '비용 분석':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = cost.ask_cost_insights(prompt)
                st.write(response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                # chat.save_chat_history(prompt, response)
                
        else:
            stream = chat.general_conversation(prompt)

            response = st.write_stream(stream)
            logger.info(f"response: {response}")

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)
        


