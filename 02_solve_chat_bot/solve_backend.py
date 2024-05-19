import os
import json
import boto3
from langchain.llms.bedrock import Bedrock
# from langchain_community.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from botocore.client import Config
from langchain.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA




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
model_kwargs_claude = {"temperature": 0, "top_k": 10, "max_tokens_to_sample": 3000}
    
    

sentences =["You are robot for 研華科技"]


def get_memory(): #create memory for this chat session
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True) #Maintains a history of previous messages
    return memory    
    
model_kwargs_claude = {"temperature": 0, "top_k": 10, "max_tokens_to_sample": 3000}
    
llm = Bedrocks(
    client=bedrock_runtime, 
    region_name='us-west-2',
    model_id="anthropic.claude-instant-v1",
    model_kwargs=model_kwargs_claude
)

### Complete this section by filling out the TODO values
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# def solve_chat_bot():
#     solve_llm = Bedrock(
#         credentials_profile_name='default',
#         model_id  = 'anthropic.claude-3-sonnet-20240229-v1:0',
#         model_kwargs = {
#             # (Required) The prompt that you want Claude to complete. For proper response generation you need to format your prompt using alternating
#             #prompt: [string]
#             "temperature": 0,
#             "top_p": 1,
#             "top_k": 250,
#             "max_tokens_to_sample": 200,
#             # Sequences that will cause the model to stop generating. (optional)
#             #"stop_sequences": [string]
#         })
#     return solve_llm
    
# def solve_chat_bot_memory():
#     llm_data = solve_chat_bot()
#     memory = ConversationBufferMemory(llm= llm_data, max_token_limit= 1000)
#     return memory

# def solve_chat_bot_converation(input_text, memory):
#     llm_chain_data = solve_chat_bot()
#     llm_conversation = ConversationChain(llm= llm_chain_data, memory = memory, verbose = True)
#     chat_reply = llm_conversation.predict(input = input_text)
#     return chat_reply    
    

# code for knowledge base
# def get_retreiver()
# retriever = AmazonKnowledgeBasesRetriever(
#   knowledge_base_id="IS2PZCVTJ9",
#    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
#)
# 應該需要在 solve_chat_bot_converation 中修改
# 前端呼叫時也要跟著修改
    