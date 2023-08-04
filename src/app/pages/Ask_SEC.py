from llama_index import Prompt
import streamlit as st
from llama_index.prompts.prompt_type import PromptType

from common.utils.misc import dict_to_named_tuple
from core import EntityIndex, SourceDocIndex

st.set_page_config(page_title="Ask SEC", page_icon="🔎")

st.title("Ask SEC")
question = st.text_area("What would you like to ask?", "")

DEFAULT_TEXT_QA_PROMPT_TMPL = """
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge,
    provided a detailed, scientific and accurate answer to the question below.
    Format the answer in markdown, and include tables, lists and links where appropriate.
    ---------------------
    {query_str}
    ---------------------
"""
DEFAULT_TEXT_QA_PROMPT = Prompt(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

if st.button("Submit"):
    if not question.strip():
        st.error(f"Please supply a question.")
    else:
        with st.spinner("Initializing..."):
            # ei = EntityIndex(model_name="ChatGPT")
            si = SourceDocIndex(model_name="GPT4")
        source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
        st.subheader("Answer from source doc index:")
        with st.spinner("Please wait..."):
            si_answer = si.query(
                question, source, prompt_template=DEFAULT_TEXT_QA_PROMPT
            )
            st.markdown(si_answer)

        # st.subheader("Answer from entity doc index:")
        # with st.spinner("Please wait..."):
        #     ei_answer = ei.query(prompt, source)
        #     st.markdown(ei_answer)
