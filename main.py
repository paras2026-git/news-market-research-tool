

import streamlit as st
from rag import process_urls, generate_answer

st.title("News and Market Research Tool")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

status_placeholder = st.sidebar.empty()
main_placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url!='']
    if len(urls) == 0:
        status_placeholder.text("You must provide at least one valid url")
    else:
        for status in process_urls(urls):
            status_placeholder.text(status)

query = main_placeholder.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    except RuntimeError:
        st.error("You must process urls first")
