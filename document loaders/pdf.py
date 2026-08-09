from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(
    r"C:\Users\win11\Desktop\Rag-project\document loaders\GRU.pdf"
)

docs = data.load()

print(len(docs))
print(docs[5].page_content)