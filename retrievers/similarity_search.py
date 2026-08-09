from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents= docs,
    embedding= embedding_model
)

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("\n=======Similarity Search Result=========\n")
similarity_result = similarity_retriever.invoke("What is Gradient Descent?")

for doc in  similarity_result:
    print(doc.page_content)
