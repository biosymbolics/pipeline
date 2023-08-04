import streamlit as st

from common.utils.misc import dict_to_named_tuple
from core import EntityIndex, SourceDocIndex

st.set_page_config(page_title="Ask SEC", page_icon="🔎")

st.title("Ask SEC")
question = st.text_area("What would you like to ask?", "")
# prompt = f"""
#     Below is a question from a technical expert in biomedicine looking to inform their drug discovery or investment strategy.
#     With that in mind, provide detailed and scientific answer to their question.
#     Format the answer in markdown, and include tables, lists and links where appropriate.
#     Here is the question:
#     {question}
# """
prompt = question

if st.button("Submit"):
    if not question.strip():
        st.error(f"Please supply a question.")
    else:
        with st.spinner("Initializing..."):
            # ei = EntityIndex(model_name="ChatGPT")
            si = SourceDocIndex(model_name="ChatGPT")  # GPT4
        source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
        st.subheader("Answer from source doc index:")
        with st.spinner("Please wait..."):
            si_answer = si.query(prompt, source)
            st.markdown(si_answer)

        # st.subheader("Answer from entity doc index:")
        # with st.spinner("Please wait..."):
        #     ei_answer = ei.query(prompt, source)
        #     st.markdown(ei_answer)
