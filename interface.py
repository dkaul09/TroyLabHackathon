import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if "generated" not in st.session_state: 
    st.session_state["generated"] = []
if "past" not in st.session_state: 
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

def get_text():
    input_text = st.text_input("You: ", st.session_state["input"], key = "input", placeholder="Your AI Consultant here. Ask me anything ...", label_visibility='hidden')
    return input_text

st.title("Consultant Bot")

api = st.sidebar.text_input("API-KEY", type="password")

 
llm = OpenAI(openai_api_key=api_key, model_name="gpt-4-0125-preview")

if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm,k=5)
conversation = ConversationChain(llm=llm, prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE, memory=st.session_state.entity_memory)

user_input = get_text()

if user_input: 
    output = conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state['generated']) -1, -1, -1):
        st.info( st.session_state["past"][i])
        st.success(st.session_state["generated"][i], icon="📈")

