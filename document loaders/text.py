from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    r"C:\Users\win11\Desktop\Rag-project\document loaders\text.txt",
    encoding="utf-8"
)

doc = loader.load()

print(doc)
print(len(doc))


print("\n",doc[0].page_content)