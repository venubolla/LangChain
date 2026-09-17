import streamlit as st
from pathlib import Path
from sqlalchemy import create_engine
import sqlite3

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_groq import ChatGroq
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQlite3 Database-Student.db", "Connect to your SQL database"]
selected_opt = st.sidebar.radio(
    label="Choose suitable option", options=radio_opt
)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host", value="localhost:3306")
    mysql_user = st.sidebar.text_input("MySQL User", value="root")
    mysql_password = st.sidebar.text_input(
        "MySQL Password", type="password"
    )
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB
    mysql_host = mysql_user = mysql_password = mysql_db = None

api_key = st.sidebar.text_input(
    label="Groq API Key", type="password"
)

if not db_uri:
    st.info("Please choose a database option to continue.")

if not api_key:
    st.info("Please add your Groq API key to continue.")


@st.cache_resource(ttl="2h")
def configure_db(
    db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None
):
    if db_uri == LOCALDB:
        db_filepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))

    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(
            create_engine(
                f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            )
        )


if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

if api_key:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-20b",
        streaming=True,
        max_tokens=800,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="tool-calling",
        handle_parsing_errors=True,
        max_iterations=5,
    )

    if "messages" not in st.session_state or st.sidebar.button(
        "Clear message history"
    ):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=False
            )
            try:
                response = agent.run(user_query, callbacks=[st_cb])
            except Exception as e:
                response = f"I ran into an issue answering that: {e}. Please try rephrasing your question."
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            st.write(response)