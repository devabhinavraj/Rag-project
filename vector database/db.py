from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


load_dotenv()

data = TextLoader(
    r"C:\Users\win11\Desktop\Rag-project\document loaders\text.txt",
    encoding= "utf-8"
)
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 300
)

chunks = splitter.split_documents(docs)

print(len(chunks))

embedding_model = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model,
    persist_directory= "chroma-db"
)


print(len(vectorstore))