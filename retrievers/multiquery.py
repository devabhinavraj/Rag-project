from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_groq import ChatGroq 
from langchain_community.document_loaders import TextLoader

load_dotenv()

# loader = TextLoader(
#     r"C:\Users\win11\Desktop\Rag-project\document loaders\text.txt",
#     encoding="utf-8"
# )

# data = loader.load()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding
)


retriever = vectorstore.as_retriever()

llm = ChatGroq(
    model= "openai/gpt-oss-120b"
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever= retriever,
    llm=llm
)

query = "What is Gradient Descent"

response = multiquery_retriever.invoke(query)

for res in response:
    print(res.page_content)
    print("\n")