from langchain_community.vectorstores import Chroma
from modules.embedder import get_embeddings

def create_vector_store(chunks, persist_dir="storage"):
    """
    Create a Chroma vector store from text chunks.
    """
    embeddings = get_embeddings()  # no arguments needed
    texts = [chunk.page_content for chunk in chunks]

    store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return store

def query_vector_store(query, vector_store, top_k=3):
    """
    Query the vector store for similar chunks.
    """
    results = vector_store.similarity_search(query, k=top_k)
    return results

