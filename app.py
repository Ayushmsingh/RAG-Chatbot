import os
import tempfile
import time
import traceback

import streamlit as st

from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import embedding_models 
from src.vectorstore import create_vectorstore
from src.rag_chain import get_rag_chain
from src.query_rewriter import get_query_rewriter, rewrite_question
from src.bm_25retriever import create_bm25_retriever
for key in (
    "vectorstore",
    "retriever",
    "bm25_retriever",
    "chain",
    "query_rewriter",
    "processed_file_name",
):
    if key not in st.session_state:
        st.session_state[key] = None
from src.hybrid_retriever import hybrid_retrieve
from src.memory import (
    initialize_memory,
    add_message,
    get_chat_history,
    get_history_as_string,
    clear_memory,
)
from src.config import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    TOP_K,
    FETCH_K,
    LAMBDA_MULT,
    BM25_K,
    HYBRID_K,
    FAISS_WEIGHT,
    BM25_WEIGHT,
    RRF_K,
    SUCCESS_MESSAGE,
    LOADING_MESSAGE,
    SEARCHING_MESSAGE,
)
from src.logger import (
    logger,
    log_pdf_uploaded,
    log_pdf_loaded,
    log_chunks_created,
    log_embedding_loaded,
    log_vectorstore_created,
    log_retriever_created,
    log_question,
    log_answer,
    log_response_time,
    log_warning,
    log_error,
    log_bm25_retriever_created,
)



# -------------------- Cached Resources --------------------
@st.cache_resource
def load_embedding_model():
    return embedding_models()


@st.cache_resource
def load_chain():
    return get_rag_chain()


@st.cache_resource
def load_query_rewriter():
    return get_query_rewriter()


# -------------------- Page Configuration --------------------
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)
initialize_memory()

st.title("📄 PDF RAG Chatbot")
st.write("Ask questions from your uploaded PDF using RAG.")

# -------------------- Session State Init --------------------
for key in (
    "vectorstore",
    "retriever",
    "chain",
    "query_rewriter",
    "processed_file_name"
):
    if key not in st.session_state:
        st.session_state[key] = None
# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        log_pdf_uploaded(uploaded_file.name)

    st.divider()

    if st.button("🗑️ Clear Chat"):
        clear_memory()
        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.chain = None
        st.session_state.processed_file_name = None
        st.rerun()

if uploaded_file is None:
    st.info("Please upload a PDF to get started.")
    st.stop()

# -------------------- Process PDF (only once per file) --------------------
if st.session_state.processed_file_name != uploaded_file.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner(LOADING_MESSAGE):
        try:
            docs = load_pdf(pdf_path)
            log_pdf_loaded(len(docs))
            if not docs:
                raise ValueError("The uploaded PDF contains no readable text.")

            chunks = split_documents(docs)
            log_chunks_created(len(chunks))
            if not chunks:
                raise ValueError("No chunks could be created from the PDF.")

            embedding_model = load_embedding_model()
            log_embedding_loaded()

            vectorstore = create_vectorstore(chunks, embedding_model)
            log_vectorstore_created()

            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": TOP_K,
                    "fetch_k": FETCH_K,
                    "lambda_mult": LAMBDA_MULT,
                },
            )
            st.session_state.bm25_retriever = create_bm25_retriever(
            chunks,
            k=BM25_K
        )
            log_bm25_retriever_created()
            log_retriever_created()

            st.session_state.chain = load_chain()
            st.session_state.query_rewriter = load_query_rewriter()
            st.session_state.processed_file_name = uploaded_file.name

        except FileNotFoundError:
            st.error("❌ The uploaded PDF could not be found.")
            st.stop()
        except PermissionError:
            st.error("❌ Permission denied while reading the PDF.")
            st.stop()
        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()
        except Exception as e:
            log_error(e)
            st.error("Unable to process the PDF.")
            st.stop()
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    logger.info("PDF processed successfully.")
    st.success(SUCCESS_MESSAGE)

# -------------------- Show Chat History --------------------
for message in get_chat_history():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------- Question Input --------------------
question = st.chat_input("Ask a question about your PDF:")

if question:
    retriever = st.session_state.retriever
    chain = st.session_state.chain
    query_rewriter = load_query_rewriter()

    # Capture history BEFORE adding the current question
    history_str = get_history_as_string()

    add_message("user", question)
    log_question(question)
    with st.chat_message("user"):
        st.markdown(question)

    answer = None
    retrieved_docs = []

    with st.spinner(SEARCHING_MESSAGE):
        try:
            start = time.time()

            # Single call — fixed duplicate/mismatched call from before
            rewritten_question = rewrite_question(
                query_rewriter, history_str, question
            )
            logger.info(f"Original Question : {question}")
            logger.info(f"Rewritten Question : {rewritten_question}")

            retrieved_docs = hybrid_retrieve(
            faiss_retriever=retriever,
            bm25_retriever=st.session_state.bm25_retriever,
            query=rewritten_question,
            k=HYBRID_K,
            faiss_weight=FAISS_WEIGHT,
            bm25_weight=BM25_WEIGHT,
            rrf_k=RRF_K,
        )

            if not retrieved_docs:
                log_warning("No relevant information found.")
                st.warning("No relevant information found in the PDF.")
                st.stop()

            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            response = chain.invoke(
                {
                    "history": history_str,
                    "context": context,
                    "question": question,
                }
            )
            answer = response.content

            end = time.time()
            log_answer()
            log_response_time(end - start)
            st.caption(f"⏱ Response Time : {end - start:.2f} sec")

        except Exception as e:
            log_error(e)
            st.exception(e)
            st.code(traceback.format_exc())
            st.stop()

    if answer:
        add_message("assistant", answer)
        with st.chat_message("assistant"):
            st.markdown(answer)

        if retrieved_docs:
            with st.expander("Retrieved Chunks"):
                for i, doc in enumerate(retrieved_docs, start=1):
                    st.markdown(f"### Chunk {i}")
                    st.write(doc.page_content)
                    st.write(f"**Metadata:** {doc.metadata}")