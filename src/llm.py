import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr

from src.config import (
    LLM_MODEL,
    TEMPERATURE,
)

load_dotenv()


def get_llm():
    """
    Returns a configured Groq LLM.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure it is set in your .env file."
        )

    return ChatGroq(
        model=LLM_MODEL,
        api_key=SecretStr(api_key),
        temperature=TEMPERATURE,
    )