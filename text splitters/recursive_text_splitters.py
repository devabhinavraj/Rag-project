from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = PyPDFLoader(
    r"C:\Users\win11\Desktop\Rag-project\document loaders\GRU.pdf"
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 20
)

chunks = splitter.split_documents(docs)

print(len(chunks))