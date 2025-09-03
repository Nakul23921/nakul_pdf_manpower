import streamlit as st
from modules.pdf_loader import load_pdf
from modules.chunker import chunk_documents
from modules.embedder import get_embeddings

from modules.vector_store import create_vector_store

from modules.chat_engine import chat_with_pdf

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with Your PDF (MiniLM Embeddings)")

# Session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    docs = load_pdf("temp.pdf")
    chunks = chunk_documents(docs)

    st.success(f"✅ Loaded {len(chunks)} chunks")

    # Create vector store
    vector_store = create_vector_store(chunks)

    st.subheader("💬 Chat")
    user_query = st.text_input("Ask a question about the PDF")

    if user_query:
        answer, sources = chat_with_pdf(user_query, vector_store, st.session_state.history)
        st.session_state.history.append((user_query, answer, sources))

# Display conversation history
if st.session_state.history:
    st.subheader("Conversation History")
    for i, (q, a, s) in enumerate(st.session_state.history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
        st.markdown(f"_Source: {s}_")
