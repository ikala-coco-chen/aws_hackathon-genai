import streamlit as st 
# import Chatbot as file
import solve_backend as solve_chat_bot
import boto3
from botocore.client import Config
import os
import streamlit as st
import langchain
from langchain.llms.bedrock import Bedrock
#from langchain.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA

def solve_chat():
    #st.title("這是title") 
    #st.write("Hi 有什麼可以幫助你的嗎?")
    
    with st.chat_message("assistant"):
        st.write("Hi! 我是數據詢問與分析機器人，有什麼可以幫助你的嗎?")
    
    
    # if 'memory' not in st.session_state: 
    #     st.session_state.memory = solve_chat_bot.solve_chat_bot_memory() #

    # # add
    # if 'chat_history' not in st.session_state: 
    #     st.session_state.chat_history = [] 

    # if 'vector_index' not in st.session_state: 
    #     with st.spinner("Indexing document..."): 
    #         st.session_state.vector_index = glib.get_index() 

    # for message in st.session_state.chat_history: 
    #     with st.chat_message(message["role"]): 
    #         st.markdown(message["text"]) 

    input_text = st.chat_input("請輸入您的問題，例如：今天的總用電量是多少?") 
    if input_text: 
    
        with st.chat_message("user"): 
            st.markdown(input_text) 
        st.session_state.chat_history.append({"role":"user", "text":input_text}) 
    
        chat_response = solve_chat_bot.solve_chat_bot_converation(input_text=input_text, memory=st.session_state.memory)
    
        with st.chat_message("assistant"): 
            st.markdown(chat_response) 
    
        st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) 



    


