import os
import boto3
import json
from langchain.llms.bedrock import Bedrock
# from langchain_community.llms import Bedrock
from langchain.memory import ConversationBufferWindowMemory
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
from botocore.client import Config
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS



REGION = "us-west-2"

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)

sentences =["You are robot name 研寶 for ADVANTECH 研華科技"]


# def normal_chat_bot():
#     normal_llm = Bedrock(
#         credentials_profile_name='default',
#         model_id  = 'anthropic.claude-3-sonnet-20240229-v1:0',
#         model_kwargs = {
#             # (Required) The prompt that you want Claude to complete. For proper response generation you need to format your prompt using alternating
#             #prompt: [string]
#             "temperature": 0.5,
#             "top_p": 1,
#             "top_k": 250,
#             "max_tokens_to_sample": 200,
#             # Sequences that will cause the model to stop generating. (optional)
#             #"stop_sequences": [string]
#         })
#     return normal_llm


    
# def normal_chat_bot_memory():
#     llm_data = normal_chat_bot()
#     memory = ConversationBufferMemory(llm= llm_data, max_token_limit= 1000)
#     return memory

# def normal_chat_bot_conversation(input_text, memory):
#     llm_chain_data = normal_chat_bot()
#     llm_conversation = ConversationChain(llm= llm_chain_data, memory = memory, verbose = True)
#     chat_reply = llm_conversation.predict(input = input_text)
#     return chat_reply   


def get_memory(): #create memory for this chat session
    
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True) #Maintains a history of previous messages
    
    return memory
    
def call_claude_sonnet(prompt):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
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


def rag_with_bedrock(query):
    # embed我們給他的sentence進model 塞進knowledge space
    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v1",
    )
    # 將文字轉為向量，再vector回
    local_vector_store = FAISS.from_texts(sentences, embeddings)

    docs = local_vector_store.similarity_search(query)
    context = ""

    for doc in docs:
        context += doc.page_content

    prompt = f"""Please answer the question of the user ask in detail.

    {context}

    Question: {query}
    Answer:"""

    return call_claude_sonnet(prompt)

