import streamlit as st
import sys
sys.path.append("./1_stock_qna")
#sys.path.append("./2_stock_query")
#sys.path.append("./3_stock_tools")
#sys.path.append("./4_stock_analysis")
sys.path.append("./01_analysis_chat_bot")
sys.path.append("./02_solve_chat_bot")
sys.path.append("./03_normal_chatbot")

from stock_qna_app import stock_qna 
# from stock_query_app import stock_query
# from stock_tools_app import stock_tools
# from stock_analysis_app import stock_analysis
from analysis_app import analysis_chat
from solve_app import solve_chat
from normal_app import normal_chat
from PIL import Image

st.set_page_config(
    page_title=" ADVANTECH Agent",
    page_icon=":🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

#st.sidebar.success("Select a a tool below.")
ff1, ff2 = st.columns([0.1, 0.9])
with ff1:
    image = Image.open('hello.png')
    st.image(image, caption='', width=60)
with ff2:
    st.title("Intelligent 智慧能源小幫手")

#page_names_to_funcs = {
#    "Stock Analysis": stock_analysis,
#    "Stock Tools": stock_tools,
#    "Stock Q&A": stock_qna, 
#    "Stock Query": stock_query
#}

page_names_to_funcs = {
    "數據詢問與分析": analysis_chat, #數據詢問與分析
    "系統操作與維修": solve_chat, #系統操作與維修
    "聊天機器人": normal_chat, #聊天機器人
    "Stock Q&A": stock_qna
}

demo_name = st.sidebar.selectbox("請選擇一個指定聊天室", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()