import os
from langchain.llms.bedrock import Bedrock
# from langchain_community.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def analysis_chat_bot():
    solve_llm = Bedrock(
        credentials_profile_name='default',
        model_id  = 'anthropic.claude-3-sonnet-20240229-v1:0',
        model_kwargs = {
            # (Required) The prompt that you want Claude to complete. For proper response generation you need to format your prompt using alternating
            #prompt: [string]
            "temperature": 0.5,
            "top_p": 1,
            "top_k": 250,
            "max_tokens_to_sample": 200,
            # Sequences that will cause the model to stop generating. (optional)
            #"stop_sequences": [string]
        })
    return solve_llm
    
def analysis_chat_bot_memory():
    llm_data = analysis_chat_bot()
    memory = ConversationBufferMemory(llm= llm_data, max_token_limit= 1000)
    return memory

def analysis_chat_bot_converation(input_text, memory):
    llm_chain_data = analysis_chat_bot()
    llm_conversation = ConversationChain(llm= llm_chain_data, memory = memory, verbose = True)
    chat_reply = llm_conversation.predict(input = input_text)
    return chat_reply    
    
        
    