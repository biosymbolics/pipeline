import streamlit as st

from common.utils.misc import dict_to_named_tuple
from core import EntityIndex, SourceDocIndex
from system import init

init()

st.title("Ask Biosymbolic.ai")
question = st.text_area("What would you like to ask?", "")
prefix = (
    "Below is a question from a technical expert in biomedicine looking to inform their drug discovery or investment strategy. "
    "With that in mind, provide detailed, scientific and comprehensive answer to their question. "
    "Format the answer in markdown, and include tables and links if appropriate. "
    "Here is the question: \n\n"
)
prompt = prefix + question

if st.button("Submit"):
    status = st.progress(0)
    if not prompt.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
            st.subheader("Answer from source doc index:")
            si = SourceDocIndex()
            si_answer = si.query(prompt, source)
            st.markdown(si_answer)
            st.subheader("Answer from entity doc index:")
            status.progress(50)
            ei = EntityIndex()
            ei_answer = ei.query(prompt, source)
            st.markdown(ei_answer)
            status.progress(50)
        except Exception as e:
            st.error(f"An error occurred: {e}")
