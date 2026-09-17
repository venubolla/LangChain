from dotenv import load_dotenv

import os
load_dotenv()
import streamlit as st

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
#Langchain Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")#Langsmith Tracing

#Prompt template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful AI Assistant. Provide me answers based on the question"),
        ("user","Question:{question}")
    ]
)

st.title("Sample LangChain GenAI App with Ollama Gemma Model")

from langchain_community.llms import Ollama

llm = Ollama(model="gemma:2b")

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
chain = prompt|llm|output_parser
input_text = st.text_input("Enter what you want to know?")

if input_text:
    st.write(chain.invoke(input_text))



