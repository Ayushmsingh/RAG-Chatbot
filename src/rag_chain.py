from pathlib import Path


from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_llm
llm = get_llm()

from src.config import LLM_MODEL, TEMPERATURE

# Load .env explicitly relative to project root, so it works
# regardless of the working directory the app is launched from.


RAG_PROMPT_TEMPLATE = """
You are an intelligent PDF assistant.

Your job is to answer questions ONLY using:

1. The conversation history.
2. The retrieved context from the uploaded PDF.

Instructions:
- Use the conversation history to understand follow-up questions.
- If the current question refers to something mentioned earlier (e.g. "it", "that", "this concept"), use the history to determine what it refers to.
- Do NOT make up information.
- If the answer cannot be found in the retrieved context, reply exactly:
"I couldn't find the answer in the provided document."

-----------------------------
Conversation History:
{history}

-----------------------------
Retrieved Context:
{context}

-----------------------------
Current Question:
{question}

Answer:
"""


def get_rag_chain():
    """Builds and returns the RAG prompt -> LLM chain."""



    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    return prompt | llm