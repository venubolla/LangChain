import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain_classic.chains import LLMMathChain, LLMChain
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import PromptTemplate

## Streamlit App

st.set_page_config(page_title="Innovative Math and Data Search Assistant Using Google Gemma2")
st.title("Innovative Math and Data Search Assistant Using Google Gemma2")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")

## Tool 1 - Wikipedia
wikipedia_wrapper = WikipediaAPIWrapper()


def wikipedia_search(query: str) -> str:
    return wikipedia_wrapper.run(query)


wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_search,
    description="A tool for searching the internet to find various information on the topics mentioned.",
)

## Tool 2 - Math tool (LLMMathChain used as a calculator)
math_chain = LLMMathChain.from_llm(llm=llm)


def calculator_run(query: str) -> str:
    # Wrapped in a plain function (instead of passing math_chain.run directly)
    # to avoid a Python 3.14 (PEP 649) incompatibility with LangChain's
    # @deprecated-decorated Chain.run method during tool-signature introspection.
    return math_chain.run(query)


calculator = Tool(
    name="Calculator",
    func=calculator_run,
    description="A tool for answering math related questions. Only input mathematical expressions need to be provided.",
)

## Tool 3 - Reasoning tool (LLMChain combining the above into general reasoning)
prompt = """
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a
detailed explanation, displayed point-wise for the question below.
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt,
)

chain = LLMChain(llm=llm, prompt=prompt_template)


def reasoning_run(query: str) -> str:
    return chain.run(query)


reasoning_tool = Tool(
    name="Reasoning tool",
    func=reasoning_run,
    description="A tool for answering logic-based and reasoning questions.",
)

## Combine all tools into an agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "👋 Welcome to the \"Math & Information Retrieval Assistant\" – your go-to tool for "
            "solving math problems and accessing information from Wikipedia. 🌎",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.text_area(
    "Enter your question:",
    placeholder="e.g. I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. "
    "Then I buy a dozen apples and 2 packs of blueberries. Each pack contains 25 blueberries. "
    "How many total pieces of fruit do I have at the end?",
)

if st.button("Find my answer"):
    if question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = assistant_agent.run(question, callbacks=[st_cb])
            except Exception as e:
                response = f"I ran into an issue answering that: {e}. Please try rephrasing your question."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            st.success(response)
    else:
        st.warning("Please enter a question.")

#it streams the agent's intermediate reasoning steps (which tool it's calling,
#  what it got back, its next thought) live into the UI as collapsible sections, 
# rather than the user only seeing a blank page until the final answer appears.
#  st.container() gives it a dedicated area on the page to render into, and 
# expand_new_thoughts=False keeps those step-by-step blocks collapsed by default (tidier UI)
#  rather than auto-expanding each one.