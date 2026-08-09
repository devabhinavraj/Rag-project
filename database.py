from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = PyPDFLoader(
    r"C:\Users\win11\Desktop\Rag-project\document loaders\deeplearning.pdf"
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model  = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model,
    persist_directory="Database"
)
