import streamlit as st

from system import initialize

initialize()

st.set_page_config(
    page_title="Hello",
    page_icon="🌐",
)

st.write("# Welcome to BioSymbolics! 🧬📈")

st.markdown(
    "This is a demo of our platform. Please select a page from the sidebar to get started."
)
