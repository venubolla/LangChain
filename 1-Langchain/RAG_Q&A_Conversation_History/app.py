## RAG Q&A Conversation with PDF including Chat History

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## set up Streamlit 
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

## Input the Groq API Key
api_key=st.text_input("Enter your Groq API key:",type="password")

## Check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="openai/gpt-oss-20b")

    ## chat interface

    session_id=st.text_input("Session ID",value="default_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    """

    session_id → lets you run multiple separate conversations in the same app 
    (e.g., different users, or you testing different topics) — each identified by a string ID.
    st.session_state.store → a dictionary that will map {session_id: conversation_history}. 
    This is initialized once and persists across Streamlit reruns

    """

    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    ## Process uploaded  PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"   
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
        """

        If you upload 2+ PDFs, each one overwrites the previous temp file before it's guaranteed to be loaded.
        In practice this usually still works because .load() runs immediately after writing, 
        but it's fragile — a safer version would use temppdf=f"./temp_{uploaded_file.name}" to give each 
        file a unique path.

        """
        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()    

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
        """
        
        create_history_aware_retriever creates a mini-pipeline that:

        Takes the chat history + the new raw question
        Sends them to the LLM with the instruction above ("rewrite this as a standalone question")
        Takes the LLM's rewritten question and runs that through retriever to fetch chunks

        Worked example:

        Input to this step	What happens	Output
        chat_history = [], input = "What is attention mechanism?"	No history yet, so the LLM just returns the question unchanged	Standalone question: "What is attention mechanism?" → chunks about attention mechanism retrieved
        chat_history = [Human: "What is attention mechanism?", AI: "..."], input = "What are its limitations?"	LLM sees "its" refers to attention mechanism from history, rewrites the question	Standalone question: "What are the limitations of the attention mechanism?" → now the retriever gets a query that actually makes sense on its own, and returns relevant chunks

        """

        # Answer question
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)


        """
        
        rag_chain.invoke({"input": "...", "chat_history": [...]})
                │
                ▼
        history_aware_retriever runs first
        → rewrites question using chat_history
        → searches vector store
        → returns: retrieved chunks
                │
                ▼
        question_answer_chain runs next
        → receives: context = [those chunks], input, chat_history
        → fills qa_prompt's {context}, chat_history, {input}
        → calls llm
        → returns: {"answer": "..."}
                │
                ▼
        rag_chain's final output:
        {
            "input": "...",
            "chat_history": [...],
            "context": [chunk1, chunk2, ...],   ← also included, so you can display sources
            "answer": "The attention mechanism has..."
        }
        
        """
        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")

