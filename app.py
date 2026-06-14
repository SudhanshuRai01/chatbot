
import streamlit as st
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import PromptTemplate

load_dotenv()
@st.cache_resource
def load_llm():
    llm=HuggingFaceEndpoint(
        repo_id="google/gemma-3n-E4B-it",
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
        max_new_tokens=512,
        temperature=0.7
    )
    return ChatHuggingFace(llm=llm)

def get_chain(llm):
    if "chain" not in st.session_state:
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=300,
            human_prefix="Human",
            ai_prefix="Assistant"
        )

        template = """You are a friendly assistant. You have a summary of the conversation so far,but you donot need to tell it.

{history}
Human: {input}
Assistant:"""

        prompt = PromptTemplate(input_variables=["history", "input"], template=template)

        st.session_state.chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False
        )

    return st.session_state.chain


st.set_page_config(page_title="Memory Chat", page_icon="🧠")

st.markdown("""
    <style>
        .block-container { max-width: 750px; padding-top: 2rem; }
        .stChatMessage { border-radius: 10px; margin-bottom: 6px; }
    </style>
""", unsafe_allow_html=True)

st.title("ChatBot")

with st.spinner("Loading model, please wait..."):
    llm = load_llm()

chain = get_chain(llm)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.predict(input=user_input)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
