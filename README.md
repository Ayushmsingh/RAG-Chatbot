# 📄 PDF RAG Chatbot

A modular **Retrieval-Augmented Generation (RAG)** chatbot that allows
users to upload a PDF and ask questions about its content.

The project combines PDF processing, document chunking, semantic search,
BM25 keyword search, hybrid retrieval, query rewriting, conversation
memory, logging, and an LLM to answer questions using information
retrieved from the uploaded document.

## 🚀 Features

-   PDF upload through a Streamlit interface
-   PDF text extraction and chunking
-   Sentence Transformer embeddings
-   FAISS semantic retrieval
-   BM25 keyword retrieval
-   Hybrid FAISS + BM25 retrieval using Reciprocal Rank Fusion
-   Query rewriting for follow-up questions
-   Conversation memory
-   Grounded RAG responses
-   Retrieved-chunk display
-   Application logging
-   Response-time tracking
-   Retrieval evaluation using Recall@5, MRR@5, and latency
-   Automatic evaluation graphs and CSV output

## 🧠 How It Works

``` text
Upload PDF
    ↓
Extract text
    ↓
Split into chunks
    ↓
Create embeddings
    ↓
Build FAISS index ──────────┐
                            │
Build BM25 index ──────────┤
                            ↓
                     Hybrid Retrieval
                            ↑
User Question → Query Rewriter
                            ↓
                   Retrieved PDF Context
                            ↓
                         Groq LLM
                            ↓
                         Answer
```

The application does not simply send the whole PDF to the LLM. It first
finds relevant pieces of the document and then gives those pieces to the
LLM as context.

## 🔎 Retrieval Methods

### FAISS

FAISS performs semantic retrieval using vector embeddings. It is useful
when the question and document express a similar idea using different
words.

### BM25

BM25 performs keyword-based retrieval. It is particularly useful when
important terms from the question appear directly in the document.

### Hybrid Retrieval

Hybrid retrieval combines FAISS and BM25. The project uses weighted
Reciprocal Rank Fusion (RRF) to combine their rankings.

The comparison is useful because the three methods can behave
differently even when they retrieve the same information.

## 🔄 Query Rewriting

The chatbot can rewrite follow-up questions into standalone questions
before retrieval.

Example:

``` text
User: What is a Transformer?

User: Explain it in simple terms.
```

The second question can be rewritten using the conversation history so
that retrieval understands what "it" refers to.

The rewriter only rewrites the question. It does not answer it.

## 🧩 Technology Stack

  Component           Technology
  ------------------- ----------------------------------------
  UI                  Streamlit
  Language            Python
  RAG Framework       LangChain
  Embeddings          sentence-transformers/all-MiniLM-L6-v2
  Vector Retrieval    FAISS
  Keyword Retrieval   BM25
  Hybrid Retrieval    FAISS + BM25 + RRF
  LLM                 Llama 3.3 70B Versatile
  LLM Provider        Groq
  Logging             Python logging
  Evaluation Graphs   Matplotlib

## 📁 Project Structure

``` text
rag/
│
├── data/
│   └── sample_data.pdf
│
├── logs/
│   └── app.log
│
├── src/
│   ├── bm_25retriever.py
│   ├── config.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── loader.py
│   ├── logger.py
│   ├── memory.py
│   ├── query_rewriter.py
│   ├── rag_chain.py
│   ├── splitter.py
│   └── vectorstore.py
│
├── app.py
├── evaluation.py
├── test.py
├── requirements.txt
├── README.md
└── .env
```

Running the evaluation creates:

``` text
evaluation_results/
├── retrieval_comparison.png
├── latency_comparison.png
├── recall_at_5.png
├── mrr_at_5.png
└── results.csv
```

## ⚙️ Setup

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag
```

### 2. Create a virtual environment

Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

If Matplotlib is not already included:

``` bash
pip install matplotlib
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

``` env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit the real API key to GitHub.

Recommended `.gitignore` entries:

``` text
.env
venv/
__pycache__/
*.pyc
evaluation_results/
```

## ▶️ Run the Application

From the project root:

``` bash
streamlit run app.py
```

Then open:

``` text
http://localhost:8501
```

Upload a PDF and ask questions about it.

## 🧪 Run Retrieval Evaluation

Run:

``` bash
python evaluation.py
```

The evaluation script:

1.  Loads the PDF.
2.  Creates document chunks.
3.  Creates embeddings.
4.  Builds the FAISS retriever.
5.  Builds the BM25 retriever.
6.  Runs FAISS, BM25, and Hybrid retrieval on the same questions.
7.  Calculates Recall@5.
8.  Calculates MRR@5.
9.  Calculates average retrieval latency.
10. Generates comparison graphs.
11. Saves the numerical results as CSV.

## 📊 Evaluation Metrics

### Recall@5

Recall@5 checks whether the relevant information appears in the top five
retrieved chunks.

Higher is better.

### MRR@5

Mean Reciprocal Rank measures how high the first relevant result
appears.

For example:

``` text
Rank 1 → 1.00
Rank 2 → 0.50
Rank 3 → 0.33
Rank 4 → 0.25
```

Higher is better.

### Average Retrieval Latency

This measures how long a retrieval method takes to return its results.

Lower is better.

## 📈 Current Evaluation Results

The current five-question evaluation produced:

  Method     Recall@5   MRR@5   Average Latency
  -------- ---------- ------- -----------------
  FAISS         1.000   1.000        0.0271 sec
  BM25          1.000   0.750        0.0007 sec
  Hybrid        1.000   1.000        0.0165 sec

These are initial results from a small evaluation set. They should not
be treated as a final benchmark.

The evaluation script automatically generates:

``` text
retrieval_comparison.png
latency_comparison.png
recall_at_5.png
mrr_at_5.png
results.csv
```

## 📝 Logging

Application logs are stored in:

``` text
logs/app.log
```

The logger records important events such as:

-   PDF upload
-   PDF loading
-   Number of chunks created
-   Embedding model loading
-   FAISS creation
-   BM25 initialization
-   Retriever initialization
-   User questions
-   Query rewriting
-   Generated answers
-   Response time
-   Errors and warnings

Example:

``` text
PDF Uploaded : sample_data.pdf
PDF Loaded Successfully | Pages : 15
Chunks Created : 93
Embedding Model Loaded
FAISS Vector Store Created
BM25 Retriever Initialized
Retriever Initialized
PDF processed successfully.
User Question : what is transformer?
Rewritten Question : What is a transformer?
Answer Generated Successfully
Response Time : 2.04 sec
```

## 🛠️ Development Status

### Completed

-   [x] PDF upload interface
-   [x] PDF loading
-   [x] Text chunking
-   [x] Sentence Transformer embeddings
-   [x] FAISS vector store
-   [x] MMR retrieval
-   [x] BM25 retrieval
-   [x] Hybrid retrieval
-   [x] Query rewriting
-   [x] Conversation memory
-   [x] RAG prompt
-   [x] Groq LLM integration
-   [x] Logging
-   [x] Response-time tracking
-   [x] Retrieved chunk display
-   [x] Retrieval evaluation
-   [x] Recall@5
-   [x] MRR@5
-   [x] Latency evaluation
-   [x] Automatic graphs
-   [x] CSV evaluation output

### Future Improvements

-   [ ] Larger evaluation dataset
-   [ ] Manually verified relevance labels
-   [ ] Retrieval parameter tuning
-   [ ] Improved chunking strategies
-   [ ] Reranking of retrieved chunks
-   [ ] Better source/citation display
-   [ ] Multiple-PDF support
-   [ ] Persistent vector storage
-   [ ] Scanned/image PDF support
-   [ ] Cloud deployment
-   [ ] Automated evaluation pipeline

## 🎯 Project Goal

The goal is to build and evaluate a practical PDF-based RAG system while
understanding how different retrieval strategies affect retrieval
quality and speed.

The main comparison is:

``` text
FAISS vs BM25 vs Hybrid
```

The project currently focuses on the retrieval stage and its evaluation
before moving to more advanced RAG improvements.


------------------------------------------------------------------------

If you find the project useful, feel free to experiment with different
PDFs, retrieval parameters, evaluation questions, and RAG components.
