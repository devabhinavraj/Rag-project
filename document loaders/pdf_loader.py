from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = PyPDFLoader(
    r"C:\Users\win11\Desktop\Rag-project\document loaders\GRU.pdf"
)

docs = data.load()

model = ChatGroq(
    model = "openai/gpt-oss-120b"
)

prompt = ChatPromptTemplate.from_messages([
    ('system' , 'You are a AI that summarizes the provided document.'),
    ('human' , '{data}'),
])

final_prompt = prompt.format_messages(
    data = docs[5].page_content
)

result = model.invoke(final_prompt)

print(result.content)