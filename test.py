from  langchain_groq import ChatGroq
from dotenv import load_dotenv
import langchain

load_dotenv()

print(langchain.__version__)

model = ChatGroq(
    model = "openai/gpt-oss-120b"
)

response = model.invoke("How are you babe?🥹")

print(response.content)