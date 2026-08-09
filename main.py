from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


loader = PyPDFLoader(
    r"C:\Users\win11\Desktop\Rag-project\document loaders\deeplearning.pdf"
)

docs = loader.load()

model = ChatGroq(
    model="openai/gpt-oss-120b"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 300
)

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate([
    ('system' , "You are an AI assistant that summarizes the provided documents clearly and concisely."),
    ('human', '{data}')
])

prompt = template.format_messages(data = chunks) 


print(len(chunks))