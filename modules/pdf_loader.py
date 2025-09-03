from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path):
    """Load PDF file into LangChain documents"""
    loader = PyPDFLoader(path)
    return loader.load()

def load_and_chunk_pdf(file, chunk_size=500, chunk_overlap=100):
    """Load and split PDF into chunks of text"""
    from PyPDF2 import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks


