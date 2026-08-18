from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from src.config import LLM_MODEL, TEMPERATURE
from src.llm import get_llm

llm = get_llm()


REWRITER_PROMPT_TEMPLATE = """
You are an expert at rewriting user questions for Retrieval-Augmented Generation (RAG).

Your ONLY task is to rewrite the latest question into a standalone question.

Rules:

1. Use the conversation history to resolve references such as:
   - it
   - this
   - that
   - they
   - these
   - those
   - he
   - she
   - them

2. Preserve the original meaning.

3. Do NOT answer the question.

4. Do NOT add any extra information.

5. If the latest question is already standalone,
   return it exactly as it is.

Conversation History:
{history}

Latest Question:
{question}

Standalone Question:
"""


def get_query_rewriter():
    """
    Returns a chain that rewrites follow-up questions
    into standalone questions using conversation history.
    """



    prompt = ChatPromptTemplate.from_template(REWRITER_PROMPT_TEMPLATE)

    return prompt | llm


def rewrite_question(chain, history: str, question: str) -> str:
    """
    Invokes the query rewriter chain and returns a standalone question.
    Falls back to the original question if rewriting fails for any reason.
    """
    try:
        response = chain.invoke(
            {
                "history": history,
                "question": question,
            }
        )
        rewritten = response.content.strip()
        return rewritten if rewritten else question
    except Exception:
        # If the rewriter fails/times out, don't block the whole pipeline —
        # just fall back to using the original question for retrieval.
        return question