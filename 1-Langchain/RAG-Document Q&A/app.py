import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## load the GROQ API Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

## If you do not have open AI key use the below Huggingface embedding
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="openai/gpt-oss-20b")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    </context>
    Question:{input}

    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## Data Ingestion step
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

"""Streamlit reruns your Python script whenever the user interacts with the UI.
#Without session state, variables could be recreated/lost.
st.session_state allows data to persist across Streamlit reruns.

st.session_state.vectors    -->> stores the FAISS vector database in the current Streamlit session.
"""

st.title("RAG Document Q&A With Groq And Lama3")

user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    """
                      User Question
                        │
                        ▼
                   Retriever
                        │
                        ▼
                  FAISS Search
                        │
                        ▼
              Relevant Documents
                        │
                        ▼
                Document Chain
                        │
                        ▼
                     Prompt
                        │
                        ▼
                    Llama LLM
                        │
                        ▼
                     Answer

    """

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    """
    
    Suppose the user asks:
        "What is the main contribution of this paper?"
        Step 1 — Question
        "What is the main contribution of this paper?"
        ↓
        Step 2 — Embedding
        The question is converted into a vector.

        Question
        ↓
        Embedding Model
        ↓
        [0.21, 0.73, 0.11, ...]
        ↓
        Step 3 — FAISS search

        FAISS finds similar document chunks.

        Question
        ↓
        FAISS
        ↓
        Chunk 17
        Chunk 43
        Chunk 52
        ...
        ↓
        Step 4 — Prompt construction

        The retrieved chunks become {context}.

        Answer based only on context.

        <context>
        Relevant chunk 1...
        Relevant chunk 2...
        Relevant chunk 3...
        </context>

        Question:
        What is the main contribution?
        ↓
        Step 5 — Llama
        The prompt is sent to Llama.
        ↓
        Step 6 — Answer
        Llama generates:
        The main contribution of the paper is...
        That becomes:
        response["answer"]
            
    """

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')

"""

Display the retrieved documents
This part is particularly useful when learning RAG:
with st.expander("Document similarity Search"):
This creates a collapsible section.
Then:
for i, doc in enumerate(response['context']):
loops through the documents retrieved by FAISS.
For every document:
st.write(doc.page_content)
displays its text.
Then:
st.write('------------------------')
creates a separator.

"""