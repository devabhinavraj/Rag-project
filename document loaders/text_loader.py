from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

loader = TextLoader(
    r"C:\Users\win11\Desktop\Rag-project\document loaders\text.txt",
    encoding="utf-8"
)

doc = loader.load()

model = ChatGroq(
    model= "openai/gpt-oss-120b"
)

template = ChatPromptTemplate.from_messages([
    ('system', 'You are a AI that summarizes the text'),

    ('human' , '{data}'),
])

final_prompt = template.format_messages(data = doc[0].page_content)

response = model.invoke(final_prompt)

print(response.content)