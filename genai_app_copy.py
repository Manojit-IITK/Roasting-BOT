import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain


# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.title("Your friendly roasting bot -- get ready to be roasted🔥")

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Build prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a roasting bot that answers in a funny but harshly honest way. Keep responses short."),
    ("user", "Previous Conversation:\n{chat_history}\n\nNow answer the user's question: {question}")
])

# Choose LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
output = StrOutputParser()

# Build chain
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=st.session_state.memory,
    output_parser=output
)

# Input
user_input = st.text_input("Ask me something")

if user_input:
    response = chain.invoke({"question": user_input})
    st.markdown(f"{response['text']}")

# Show history
st.divider()
st.subheader("Chat History")

if st.session_state.memory.chat_memory.messages:
    for msg in st.session_state.memory.chat_memory.messages:
        if msg.type == "human":
            st.markdown(f"**You:** {msg.content}")
        else:
            st.markdown(f"**Bot:** {msg.content}")
else:
    st.info("No chat history yet. Start chatting!")

# Clear button
if st.button("Clear Chat History"):
    st.session_state.memory.clear()
    st.success("Chat history cleared!")
