from dotenv import load_dotenv

import os
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-120b",api_key=groq_api_key
)

from langchain_core.prompts import ChatPromptTemplate

generic_template = "Translate the following from English to {language}"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",generic_template),
        ("user","{text}")
    ]
)

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = prompt|model|parser

"""FastAPI is a Python web framework used to build HTTP APIs. It is built on:

Starlette: handles HTTP requests, routing, middleware, and ASGI communication.
Pydantic: validates request data and generates JSON schemas.
Uvicorn: runs the application as a web server."""

from fastapi import FastAPI

app = FastAPI(title="Langchain Server",
              version="1.0",
              description="English to other language translator")

from langserve import add_routes

"""This imports LangServe’s route-registration helper.

LangServe connects a LangChain Runnable, such as your LCEL chain, to FastAPI. 
It creates HTTP endpoints around the chain."""

add_routes(app,chain,path="/chain")

"""This tells LangServe:

Register routes on app
Use chain as the backend operation
Put the routes below /chain
"""

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)


"""

HTTP request
    ↓
FastAPI route
    ↓
LangServe request validation
    ↓
chain.invoke(...)
    ↓
prompt
    ↓
ChatGroq model
    ↓
StrOutputParser
    ↓
JSON response

"""