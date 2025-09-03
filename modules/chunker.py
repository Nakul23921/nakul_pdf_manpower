from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    """
    Takes a list of Document objects and splits them into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)





