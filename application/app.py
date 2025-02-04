import streamlit as st 
import chat

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
    "Agent (Reflection)": [
        "Reflection Workflow를 수행하는 Agent 구현합니다."
    ],
    "Agent (Planning)": [
        "Planning Workflow를 수행하는 Agent 구현합니다."
    ],
    "Agent (Multi-agent Collaboration)": [
        "Planning/Reflection agent들을 이용하여 Multi-agent Collaboration Workflow을 수행합니다. 여기서 Reflection agent들은 병렬처리하여 수행시간을 단축합니다."
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
        label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "RAG", "Agent (Tool Use)", "Agent (Reflection)", "Agent (Planning)", "Agent (Multi-agent Collaboration)"], index=0
    )   
    st.info(mode_descriptions[mode][0])    
    # print('mode: ', mode)

    # model selection box
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ('Nova Pro', 'Nova Lite', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku')
    )

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    #print('debugMode: ', debugMode)

    # multi region check box
    select_multiRegion = st.checkbox('Multi Region', value=False)
    multiRegion = 'Enable' if select_multiRegion else 'Disable'
    #print('multiRegion: ', multiRegion)

    # chart checkbox 
    selected_chart = st.checkbox('Chart', value=True)
    chart = 'Enable' if selected_chart else 'Disable'
    #print('chart: ', chart)

    chat.update(modelName, debugMode, multiRegion)

    st.subheader("📋 문서 업로드")
    # print('fileId: ', chat.fileId)
    uploaded_file = st.file_uploader("RAG를 위한 파일을 선택합니다.", type=["pdf", "txt", "py", "md", "csv", "json"], key=chat.fileId)

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

# Preview the uploaded image in the sidebar
file_name = ""
if uploaded_file is not None and clear_button==False:
    print("uploaded_file.name: ", uploaded_file.name)
    print("CSAT_evaluator: ", CSAT_evaluator)

    if uploaded_file.name and CSAT_evaluator=="Disable":
        chat.initiate()

        if debugMode=='Enable':
            status = '선택한 파일을 업로드합니다.'
            print('status: ', status)
            st.info(status)

        file_name = uploaded_file.name
        file_url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        print('file_url: ', file_url) 

        chat.sync_data_source()  # sync uploaded files
            
        status = f'선택한 "{file_name}"의 내용을 요약합니다.'
        # my_bar = st.sidebar.progress(0, text=status)
        
        # for percent_complete in range(100):
        #     time.sleep(0.2)
        #     my_bar.progress(percent_complete + 1, text=status)
        if debugMode=='Enable':
            print('status: ', status)
            st.info(status)
    
        msg = chat.get_summary_of_uploaded_file(file_name, st)
        st.session_state.messages.append({"role": "assistant", "content": f"선택한 문서({file_name})를 요약하면 아래와 같습니다.\n\n{msg}"})    
        print('msg: ', msg)
        st.rerun()

    elif uploaded_file.name and CSAT_evaluator == "Enable" and uploaded_file.name.lower().endswith((".json")): # csv only   
        guide = "CSAT Evaluation을 시작합니다."
        st.write(guide)
        st.session_state.messages.append({"role": "assistant", "content": guide})
        state_of_CSAT_evaluator = True

        chat.solve_CSAT_problem(uploaded_file.getvalue(), st)

# print("state_of_CSAT_evaluator: ", state_of_CSAT_evaluator)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
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
    elif mode == 'Agent (Multi-agent Collaboration)':
        col1, col2, col3 = st.columns([0.1, 2.0, 0.1])    
        url = "https://raw.githubusercontent.com/kyopark2014/agentic-workflow/main/contents/multi_agent_collaboration.png"
        col2.image(url)

# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    print('prompt: ', prompt)
    
    with st.chat_message("assistant"):
        if mode == '일상적인 대화':
            stream = chat.general_conversation(prompt)
            response = st.write_stream(stream)
            print('response: ', response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            # st.rerun()

            chat.save_chat_history(prompt, response)

        elif mode == 'RAG':
            with st.status("running...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_rag_with_knowledge_base(prompt, st)                           
                st.write(response)
                print('response: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Agent (Tool Use)':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_agent_executor(prompt, st)
                st.write(response)
                print('response: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 
        
        elif mode == 'Agent (Reflection)':
            with st.status("thinking...", expanded=True, state="running") as status:
                # esponse, reference_docs = chat.run_knowledge_guru(prompt, st)
                response, reference_docs = chat.run_reflection(prompt, st)     
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Agent (Planning)':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_planning(prompt, st)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Agent (Multi-agent Collaboration)':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_long_form_writing_agent(prompt, st)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        else:
            stream = chat.general_conversation(prompt)

            response = st.write_stream(stream)
            print('response: ', response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)
        


