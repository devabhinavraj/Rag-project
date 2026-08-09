from dotenv import  load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
url = "https://www.apple.com/in/macbook-pro/"

data = WebBaseLoader(url)

docs= data.load()
print(docs)