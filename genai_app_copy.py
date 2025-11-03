import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Get API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Streamlit UI
st.title("Your friendly roasting bot -- get ready to be roasted🔥")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Build prompt with message placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a roasting bot that answers in a funny but harshly honest way. Keep responses short."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
])

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Build chain using LCEL
chain = prompt | llm | StrOutputParser()

# Input
user_input = st.text_input("Ask me something", key="user_input")

if user_input:
    # Invoke chain with chat history
    response = chain.invoke({
        "question": user_input,
        "chat_history": st.session_state.chat_history
    })
    
    # Add messages to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
    
    # Display response
    st.markdown(f"{response}")

# Show history
st.divider()
st.subheader("Chat History")

if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        else:
            st.markdown(f"**Bot:** {msg.content}")
else:
    st.info("No chat history yet. Start chatting!")

# Clear button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")
    st.rerun()

