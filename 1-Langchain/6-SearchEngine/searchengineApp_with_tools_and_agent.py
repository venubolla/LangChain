import streamlit as st
import arxiv as arxiv_sdk
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent,AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_core.tools import Tool
import os
from dotenv import load_dotenv

import wikipedia
wikipedia.set_user_agent("SearchEngineApp/1.0 (contact: your-email@example.com)")


## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_query=ArxivQueryRun(api_wrapper=arxiv_wrapper)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_arxiv_search(query: str) -> str:
    """Avoid repeatedly calling arXiv for the same paper query."""
    return arxiv_query.run(query)

def safe_arxiv_search(query: str) -> str:
    """Return a tool observation instead of crashing when arXiv is unavailable."""
    try:
        return cached_arxiv_search(query)
    except arxiv_sdk.HTTPError:
        return (
            "arXiv is temporarily unavailable or rate-limiting this request. "
            "Do not retry arXiv immediately; use the Search tool as a fallback."
        )

arxiv=Tool(
    name="arxiv",
    description=arxiv_query.description,
    func=safe_arxiv_search,
)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

#The for loop replays the full conversation so far, rendering each message as a chat bubble 
#(st.chat_message gives you the little avatar + styled bubble UI, similar to ChatGPT's interface).

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="qwen/qwen3.6-27b",streaming=True,max_tokens=800)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(prompt,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

#st.chat_input(...) renders the text box at the bottom of the page (the actual chat input widget).
#  It returns None until the user submits something.
#   prompt := ... is Python's walrus operator — it assigns the input to prompt and evaluates 
# truthiness in the same line, so this block only runs once the user actually types and submits a message.
#The new message is appended to history and immediately rendered as a user chat bubble.

# with st.chat_message("assistant"): opens an assistant-style chat bubble container to render everything inside it.
# StreamlitCallbackHandler(st.container(), expand_new_thoughts=False) — this is the standout feature of the app. 
# As the agent works through its ReAct loop (e.g. "I should search arXiv for this" → calls arxiv tool → gets an observation → "Now I should search the web too" → calls search tool → ... → final answer), 
# each step gets streamed live into the UI as collapsible sections.
