from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="Database",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        'k':3,
        'fetch_k':10,
        'lambda_mult' : 0.5 
    }
)

llm = ChatGroq(
    model= "openai/gpt-oss-120b"
)

template = ChatPromptTemplate(
    [
        (
            'system' ,
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."

"""
        ),
        (
            'human' ,
            """Context:
{context}

Question:
{question}
"""
        ),
    ]
)


print("press 0 to exit ")
while True:
    query = input("USER:")
    if query == "0":
        print("EXIT!!")
        break

    docs = retriever.invoke(query)

    context = "\n".join(
        [docs.page_content for docs in docs]
    )

    final_prompt = template.invoke(
        {
            'context': context,
            'question' : query
        }
    )

    response = llm.invoke(final_prompt)
    print(f"\n AI: {response.content}")
