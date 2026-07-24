import streamlit as st
from functions import (
    load_and_chunk_document,
    create_embeddings,
    search_chunks,
    generate_answer
)
st.set_page_config(page_title="RAG Document QA System", layout="wide")
st.title("📚 RAG Question-Answering System")
st.sidebar.header("Configuration")
cohere_api_key = st.sidebar.text_input(
    "Cohere API Key", 
    type="password", 
    help="Enter your Cohere API key for answer generation"
)
k_value = st.sidebar.slider("Number of Chunks (k)", min_value=1, max_value=5, value=3)

st.header("1. Upload Document")
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("Processing document and generating embeddings..."):
        chunks = load_and_chunk_document(uploaded_file)
        embeddings = create_embeddings(chunks)
        
    st.success(f"Document processed successfully! Created {len(chunks)} text chunks.")

    st.header("2. Ask a Question")
    query = st.text_input("Enter your question about the document:")
    
    search_button = st.button("Search & Answer")

    if search_button and query:
        with st.spinner("Searching for relevant chunks..."):
            top_k_chunks = search_chunks(query, chunks, embeddings, k=k_value)

        st.subheader("🔍 Top Relevant Chunks")
        for idx, chunk in enumerate(top_k_chunks, start=1):
            with st.expander(f"Chunk {idx}"):
                st.write(chunk)

        if cohere_api_key:
            with st.spinner("Generating answer with Cohere..."):
                try:
                    answer = generate_answer(query, top_k_chunks, cohere_api_key)
                    st.subheader("🤖 Generated Answer")
                    st.success(answer)
                except Exception as e:
                    st.error(f"Error generating answer from Cohere: {str(e)}")
        else:
            st.warning(" Please provide a Cohere API key in the sidebar to generate AI answers.")
else:
    st.info("Please upload a PDF or TXT document to begin.")