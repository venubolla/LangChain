import streamlit as st
import validators

from langchain_groq import ChatGroq
from langchain_classic.chains import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website")
st.title("🦜 LangChain: Summarize Text From YT Video or Website")

## Sidebar for Groq API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API key", value="", type="password")

st.subheader("Summarize URL")
generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="Paste a YouTube or website URL here")

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


if st.button("Summarize"):
    if not groq_api_key.strip():
        st.error("Please provide your Groq API key.")
    elif not generic_url.strip():
        st.error("Please provide a URL to summarize.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
    else:
        try:
            with st.spinner("Loading content and summarizing..."):
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="openai/gpt-oss-120b",
                    max_tokens=800,
                )

                if is_youtube_url(generic_url):
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, add_video_info=False
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (SummarizerApp/1.0)"
                        },
                    )

                docs = loader.load()

                if not docs or not docs[0].page_content.strip():
                    st.error(
                        "Could not extract any content from that URL. "
                        "The video may have no transcript, or the site may be blocking access."
                    )
                else:
                    chain = load_summarize_chain(
                        llm=llm, chain_type="stuff", prompt=prompt
                    )
                    output_summary = chain.run(docs)
                    st.success(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")