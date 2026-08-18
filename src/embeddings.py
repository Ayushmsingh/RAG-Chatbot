from langchain_huggingface import HuggingFaceEmbeddings

def embedding_models():
    embedding = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding