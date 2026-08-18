import logging
import os
from logging.handlers import RotatingFileHandler

# ==========================================================
# LOG CONFIGURATION
# ==========================================================

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================================
# CREATE LOGGER
# ==========================================================

logger = logging.getLogger("PDF_RAG_Chatbot")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers on Streamlit reruns
if not logger.handlers:

    # ---------------- File Handler ----------------
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8"
    )

    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # ---------------- Console Handler ----------------
    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def log_pdf_uploaded(filename):
    logger.info(f"PDF Uploaded : {filename}")


def log_pdf_loaded(num_pages):
    logger.info(f"PDF Loaded Successfully | Pages : {num_pages}")


def log_chunks_created(num_chunks):
    logger.info(f"Chunks Created : {num_chunks}")


def log_embedding_loaded():
    logger.info("Embedding Model Loaded")


def log_vectorstore_created():
    logger.info("FAISS Vector Store Created")


def log_retriever_created():
    logger.info("Retriever Initialized")


def log_question(question):
    logger.info(f"User Question : {question}")


def log_answer():
    logger.info("Answer Generated Successfully")


def log_response_time(seconds):
    logger.info(f"Response Time : {seconds:.2f} sec")


def log_warning(message):
    logger.warning(message)


def log_error(error):
    logger.exception(error)

def log_bm25_retriever_created():
    logger.info("BM25 Retriever Initialized")