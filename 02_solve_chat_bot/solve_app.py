import boto3
from botocore.client import Config
import os
import json
import streamlit as st
import langchain
from langchain.llms.bedrock import Bedrock
from langchain.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
from langchain import LLMChain
from langchain.memory import ConversationBufferWindowMemory
# abc
def call_claude_sonnet(question,prompt):
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )
    prompt = f"""Please answer the question of the user ask in detail.
        Question: {question}
        Answer:"""

    prompt_config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system":"You are 研寶, an ADVANTECH AI assistant.\
                 Your goal is to provide informative and substantive responses with both Tradionnal Chinese and English \
                 to queries",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
    
    body = json.dumps(prompt_config)
    
    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"
    
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    
    results = response_body.get("content")[0].get("text")
    return results




def get_memory(): #create memory for this chat session
    
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True) #Maintains a history of previous messages
    
    return memory

def solve_chat():
    
    with st.chat_message("assistant"):
        st.write("Hi! 我是系統操作與維修機器人，有什麼可以幫助你的嗎?")
    
    REGION = "us-west-2"

    # Setup bedrock
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION,
    )
    
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="IS2PZCVTJ9",
        retrieval_config={
            "vectorSearchConfiguration": {"numberOfResults": 4}},
    )
    # model_kwargs_claude = {
    #     "message": "\n\nHuman: You are 研寶, an ADVANTECH AI assistant.\
    #         Your goal is to provide informative and substantive responses with both Tradionnal Chinese and English \
    #         to queries \n\nAssistant:",
    #     "system": 
    #         "You are 研寶, an ADVANTECH AI assistant.\
    #         Your goal is to provide informative and substantive responses with both Tradionnal Chinese and English \
    #         to queries", 
    #     "temperature": 0,
    #     "top_p": 1,
    #     "top_k": 250,
    #     "max_tokens_to_sample": 2000,
    # }

    model_kwargs_claude = {
        "max_tokens": 1024, 
        "system": 
            "Your name is 研寶, an ADVANTECH AI assistant.\
            Your goal is to provide informative and substantive responses with both Tradionnal Chinese and English \
            to queries", 
        "messages": [{"role": "user", "content": "Hello, 研寶"}], 
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": 0,
        "top_p": 1,
        "top_k": 250,
    }

    
    
    llm = BedrockChat(
        client=bedrock_runtime, 
        region_name='us-west-2',
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs=model_kwargs_claude
    )
    # 檢索型問答
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    

    
    if 'memory' not in st.session_state: 
        st.session_state.memory = get_memory()

    # add
    if 'chat_history' not in st.session_state: 
        st.session_state.chat_history = [] 
    
    # if "messages" not in st.session_state:
    #     st.session_state.messages = []
    
    for message in st.session_state.chat_history: 
        with st.chat_message(message["role"]): 
            st.markdown(message["text"])
            
    question = st.chat_input("Ask me a question")

    if question:
        
        with st.chat_message("Human"):
            st.markdown(question)
            
        st.session_state.chat_history.append({"role": "Human", "text": question})
        
        with st.chat_message("assistant"):
            with st.spinner('Processing...'):
                message_placeholder = st.empty()
                full_response = ""
                answer = qa(question)
                full_response = answer['result']
                final_response = call_claude_sonnet(question,full_response)
                message_placeholder.markdown(final_response)
                
        st.session_state.chat_history.append({"role": "assistant", "text": final_response})

    