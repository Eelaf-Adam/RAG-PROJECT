"""
rag.py
======
A CPU/GPU-compatible Retrieval-Augmented Generation (RAG) pipeline built with:
  - LangChain       : orchestration framework
  - FAISS           : vector similarity search
  - Sentence Transformers (BAAI/bge-large-en-v1.5) : document embeddings
  - TinyLlama-1.1B  : open-source language model

Designed to run on Google Colab (free T4 GPU or CPU fallback).

Author : <your name>
Date   : 2025
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Central configuration — change values here to customise the pipeline."""
    DOCUMENT_PATH        = "your_document.pdf"
    CHUNK_SIZE           = 500
    CHUNK_OVERLAP        = 50
    EMBEDDING_MODEL      = "BAAI/bge-large-en-v1.5"
    LLM_MODEL            = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    MAX_NEW_TOKENS       = 512
    TEMPERATURE          = 0.3
    REPETITION_PENALTY   = 1.15
    TOP_K                = 5
    FETCH_K              = 20
    SIMILARITY_THRESHOLD = 0.5
    DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# DOCUMENT LOADING & CHUNKING
# ==============================================================================

def load_and_split(file_path, chunk_size, chunk_overlap):
    """Load a PDF or text file and split it into overlapping chunks."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    print(f"[Loader] Loaded {len(documents)} page(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"[Splitter] Created {len(chunks)} chunks")
    return chunks


# ==============================================================================
# VECTOR STORE
# ==============================================================================

def build_vectorstore(chunks, embedding_model_name, device):
    """Convert chunks into embeddings and store them in FAISS."""
    print(f"[Embeddings] Loading {embedding_model_name} on {device}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("[FAISS] Building vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[FAISS] Vector store ready.")
    return vectorstore, embeddings


# ==============================================================================
# LANGUAGE MODEL
# ==============================================================================

def load_llm(model_id, device, max_new_tokens, temperature, repetition_penalty):
    """Load TinyLlama and wrap it in a LangChain HuggingFacePipeline."""
    print(f"[LLM] Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=repetition_penalty
    )
    print("[LLM] Model ready.")
    return HuggingFacePipeline(pipeline=text_pipeline)


# ==============================================================================
# PROMPT TEMPLATE
# ==============================================================================

def build_prompt():
    """Build a prompt template using TinyLlama ChatML format."""
    template = """<|system|>
You are a helpful assistant. Answer using only the provided context.
If the answer is not in the context, say "I do not have enough information."</s>
<|user|>
Context:
{context}

Question: {question}</s>
<|assistant|>"""
    return PromptTemplate(template=template, input_variables=["context", "question"])


# ==============================================================================
# RETRIEVER
# ==============================================================================

def build_retriever(vectorstore, embeddings, top_k, fetch_k, similarity_threshold):
    """Build an MMR retriever with a similarity filter."""
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": fetch_k}
    )
    compressor = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=similarity_threshold
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )


# ==============================================================================
# RAG CHAIN
# ==============================================================================

def build_rag_chain(llm, retriever, prompt):
    """Assemble the full RAG chain."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


# ==============================================================================
# QUERY INTERFACE
# ==============================================================================

def ask(rag_chain, question):
    """Ask a question and print the answer with sources."""
    print(f"\nQuestion : {question}")
    print("-" * 60)
    result = rag_chain.invoke({"query": question})
    print(f"Answer   : {result['result']}")
    print("\nSources used:")
    for i, doc in enumerate(result["source_documents"]):
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content[:150].replace("\n", " ")
        print(f"  [{i+1}] Page {page}: {snippet}...")


def chat_loop(rag_chain):
    """Start an interactive question-answering loop."""
    print("\nRAG Chatbot ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        ask(rag_chain, user_input)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    cfg = Config()
    print("=" * 60)
    print("  RAG Pipeline — TinyLlama + FAISS + BGE Embeddings")
    print(f"  Device : {cfg.DEVICE}")
    print("=" * 60)

    chunks                  = load_and_split(cfg.DOCUMENT_PATH, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    vectorstore, embeddings = build_vectorstore(chunks, cfg.EMBEDDING_MODEL, cfg.DEVICE)
    llm                     = load_llm(cfg.LLM_MODEL, cfg.DEVICE, cfg.MAX_NEW_TOKENS, cfg.TEMPERATURE, cfg.REPETITION_PENALTY)
    prompt                  = build_prompt()
    retriever               = build_retriever(vectorstore, embeddings, cfg.TOP_K, cfg.FETCH_K, cfg.SIMILARITY_THRESHOLD)
    rag_chain               = build_rag_chain(llm, retriever, prompt)

    chat_loop(rag_chain)


if __name__ == "__main__":
    main()
