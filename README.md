# RAG Pipeline — Chat with Your Documents

A lightweight, fully open-source Retrieval-Augmented Generation (RAG) system
that lets you ask questions about any PDF or text document.
Built to run on Google Colab's free T4 GPU (or CPU fallback) with no paid APIs.

---

## What is RAG?

RAG combines two ideas:
1. **Retrieval** — Search your document for the most relevant passages
2. **Generation** — Feed those passages to a language model to produce a grounded answer

This prevents the model from hallucinating by anchoring its answers in your actual documents.

---
## Notebook

You can explore the implementation here:  
[Open in Google Colab](https://colab.research.google.com/drive/1X-vtxP0qMDdKC42wEFaKWJhoYVXfwwKv?usp=sharing)
---


## Stack

| Component          | Tool                       | Why                             |
|--------------------|----------------------------|---------------------------------|
| Orchestration      | LangChain                  | Connects all components cleanly |
| Embeddings         | BAAI/bge-large-en-v1.5     | Top-ranked on MTEB leaderboard  |
| Vector search      | FAISS (GPU)                | Fast, free, runs locally        |
| Language model     | TinyLlama-1.1B-Chat        | ~600MB, no quantization needed  |
| Retrieval strategy | MMR + similarity filter    | Diverse, relevant chunks only   |

---

## Quickstart (Google Colab)

### 1. Install dependencies
```python
!pip install langchain langchain-community
!pip install sentence-transformers
!pip install faiss-gpu
!pip install transformers accelerate
!pip install pypdf
```

### 2. Clone the repo
```python
!git clone https://github.com/<your-username>/rag-pipeline.git
%cd rag-pipeline
```

### 3. Upload your document
```python
from google.colab import files
uploaded = files.upload()
```

### 4. Run
```python
from rag import *

cfg = Config()
cfg.DOCUMENT_PATH = "your_file.pdf"

chunks                  = load_and_split(cfg.DOCUMENT_PATH, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
vectorstore, embeddings = build_vectorstore(chunks, cfg.EMBEDDING_MODEL, cfg.DEVICE)
llm                     = load_llm(cfg.LLM_MODEL, cfg.DEVICE, cfg.MAX_NEW_TOKENS, cfg.TEMPERATURE, cfg.REPETITION_PENALTY)
prompt                  = build_prompt()
retriever               = build_retriever(vectorstore, embeddings, cfg.TOP_K, cfg.FETCH_K, cfg.SIMILARITY_THRESHOLD)
rag_chain               = build_rag_chain(llm, retriever, prompt)

chat_loop(rag_chain)
```

---

## Configuration

All settings live in the `Config` class in `rag.py`:

| Setting               | Default                       | Description                 |
|-----------------------|-------------------------------|-----------------------------|
| DOCUMENT_PATH         | your_document.pdf             | Path to your file           |
| CHUNK_SIZE            | 500                           | Characters per chunk        |
| CHUNK_OVERLAP         | 50                            | Overlap between chunks      |
| EMBEDDING_MODEL       | BAAI/bge-large-en-v1.5        | Embedding model             |
| LLM_MODEL             | TinyLlama/TinyLlama-1.1B-Chat | Language model              |
| MAX_NEW_TOKENS        | 512                           | Max tokens per response     |
| TEMPERATURE           | 0.3                           | Lower = more factual        |
| TOP_K                 | 5                             | Chunks retrieved per query  |
| SIMILARITY_THRESHOLD  | 0.5                           | Minimum relevance score     |

---

## Project Structure
```
rag-pipeline/
├── rag.py      # Full pipeline: loading, embeddings, LLM, retrieval, chat
└── README.md   

```

---

## License

MIT — free to use, modify, and distribute.
