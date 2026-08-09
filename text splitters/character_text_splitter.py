from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq

load_dotenv()

data = TextLoader(
    r"C:\Users\win11\Desktop\Rag-project\document loaders\notes.txt",
    encoding= "utf-8"
)

docs = data.load()

model = ChatGroq(
    model=  "openai/gpt-oss-120b"
)

text_splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 10,
    chunk_overlap = 1
)

chunk = text_splitter.split_documents(docs)

print(len(chunk))

print(chunk[0].page_content)
