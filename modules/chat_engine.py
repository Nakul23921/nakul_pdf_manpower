def chat_with_pdf(query, vector_store, history=None, k=3):
    """Query the PDF via vector store and return answer with sources"""
    results = vector_store.similarity_search(query, k=k)

    if not results:
        return "Sorry, I couldn’t find relevant info in the PDF.", []

    # Combine results into an answer
    answer = "\n".join([r.page_content for r in results])
    sources = [r.metadata for r in results]

    return answer, sources

