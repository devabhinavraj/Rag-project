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

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

print("\n=======MMR Result=========\n")
mmr_result = mmr_retriever.invoke("What is Gradient Descent?")

for doc in  mmr_result:
    print(doc.page_content)
