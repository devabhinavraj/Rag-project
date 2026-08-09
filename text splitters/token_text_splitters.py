from dotenv import load_dotenv
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

data = PyPDFLoader(
    r"C:\Users\win11\Desktop\Rag-project\document loaders\GRU.pdf",
    
)
docs = data.load()



splitter = TokenTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)

print(len(chunks))