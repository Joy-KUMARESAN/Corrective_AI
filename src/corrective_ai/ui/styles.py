import streamlit as st

CSS = """
<style>
body, .stApp { background-color: #f6f7fa; color: #222; }
.stTextInput>div>div>input { background-color: #f6f7fa; color: #222; }
.stDataFrame { background-color: #f6f7fa; }
.chat-bubble-bot {
    background: #e9ecef; color: #222; border-radius: 8px; padding: 10px;
    margin-bottom: 8px; margin-right: 35%;
    border-left: 6px solid #2b7cff;
}
.chat-bubble-user {
    background: #2b7cff; color: #fff; border-radius: 8px; padding: 10px;
    margin-bottom: 8px; margin-left: 35%; text-align: right;
}
.example-box {
    background: #f1f5fb; border-radius: 6px; padding: 10px 14px;
    margin-bottom: 10px; font-size: 0.96em; border-left: 4px solid #2b7cff;
}
</style>
"""

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
