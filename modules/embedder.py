from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize embeddings model once
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings():
    """
    Return the initialized embeddings object.
    """
    return embedding_model

