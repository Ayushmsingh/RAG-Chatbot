from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks, embedding_model):
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vectorstore