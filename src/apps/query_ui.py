import os
import streamlit as st
from dotenv import load_dotenv
import logging

from common.utils.misc import dict_to_named_tuple
from core import EntityIndex, SourceDocIndex

# Load environment variables from .env file
load_dotenv("/Users/kristinlindquist/development/pipeline/.env")

api_key = os.environ.get("PINECONE_API_KEY")

logging.getLogger().setLevel(logging.INFO)

# Define a simple Streamlit app
st.title("Ask Biosymbolic.ai")
question = st.text_input("What would you like to ask?", "")
prefix = (
    "Below is a question asked by a knowledgeable and technical investor "
    "looking to gain knowledge to inform investments and IP acquisition. "
    "Provide a detailed and specific answer to their question. "
    "Format the answer in markdown. \n\n"
    "For example, if the user asks 'What compounds are in Pfizer's pipeline?' "
    "reply with a list of the compounds and salient information about each. \n\n"
    "Here is the question: \n\n"
)
prompt = prefix + question

# If the 'Submit' button is clicked
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
