import streamlit as st


def initialize_memory():
    """
    Initialize chat history.
    """

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def add_message(role, message):
    """
    Add a message to chat history.
    """

    st.session_state.chat_history.append(
        {
            "role": role,
            "content": message
        }
    )


def get_chat_history():
    """
    Return the complete chat history.
    """

    return st.session_state.chat_history


def clear_memory():
    """
    Clear conversation.
    """

    st.session_state.chat_history = []

def get_history_as_string():

    history = ""

    for msg in st.session_state.chat_history:
        history += f"{msg['role'].capitalize()}: {msg['content']}\n"

    return history