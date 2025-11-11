import streamlit as st

def chat_bubble(sender: str, message: str):
    if sender == "bot":
        st.markdown(f'<div class="chat-bubble-bot">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-user">{message}</div>', unsafe_allow_html=True)
