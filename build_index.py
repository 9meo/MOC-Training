#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
สร้าง FAISS index แบบกระชับ
"""

import os
import faiss
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# Config
DATA_DIR = "data"
OUTPUT_DIR = "storage"
FAISS_PATH = f"{OUTPUT_DIR}/faiss_index.faiss"

def main():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    embed_model = HuggingFaceEmbedding("BAAI/bge-m3", normalize=True)
    
    # Load & chunk documents
    print("Loading documents...")
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    nodes = TokenTextSplitter(chunk_size=128, chunk_overlap=16).get_nodes_from_documents(docs)
    print(f"Created {len(nodes)} chunks from {len(docs)} documents")
    
    # Create FAISS index
    print("Building FAISS index...")
    faiss_index = faiss.IndexFlatIP(1024)  # bge-m3 dimension
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(nodes, embed_model=embed_model, storage_context=storage_context)
    
    # Save
    print("Saving...")

    index.storage_context.persist(OUTPUT_DIR)
    
    print(f"✅ Done! Index saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()