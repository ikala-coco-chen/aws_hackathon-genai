import streamlit as st 
# import Chatbot as file
import normal_backend as normal_chat_bot

# def print_result(st, chat_response):
#     st.markdown(chat_response['intermediate_steps'][1])
#     if not chat_response['result']:
#         st.markdown('Result: No result')
#     else:
#         st.markdown('Result:' + chat_response['result']) 

def normal_chat():
    #st.title("這是title") 
    #st.write("Hi 有什麼可以幫助你的嗎?")
    
    with st.chat_message("assistant"):
        st.write("Hi! 有什麼可以幫助你的嗎?")
    
    if 'memory' not in st.session_state: 
        st.session_state.memory = normal_chat_bot.get_memory()

    # add
    if 'chat_history' not in st.session_state: 
        st.session_state.chat_history = [] 

    # if 'vector_index' not in st.session_state: 
    #     with st.spinner("Indexing document..."): 
    #         st.session_state.vector_index = glib.get_index() 

    for message in st.session_state.chat_history: 
        with st.chat_message(message["role"]): 
            st.markdown(message["text"]) 

    input_text = st.chat_input("Hi! 有什麼可以幫助你的嗎?") 
    if input_text: 
        
        with st.chat_message("user"): #display a user chat message
            st.markdown(input_text) #renders the user's latest message
        
        st.session_state.chat_history.append({"role":"user", "text":input_text}) #append the user's latest message to the chat history
        
        chat_response = normal_chat_bot.rag_with_bedrock(input_text)
    
        with st.chat_message("assistant"): #display a bot chat message
            with st.spinner('Processing...'):
                st.markrag_with_bedrockdown(normal_chat_bot.rag_with_bedrock(input_text))
            #print_result(st, chat_response)
    
        st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) #append the bot's latest message to the chat history


    


