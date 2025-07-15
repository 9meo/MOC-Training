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
from llama_index.core import Settings
# Config
DATA_DIR = "data"
OUTPUT_DIR = "storage"


def main():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    embed_model = HuggingFaceEmbedding("BAAI/bge-m3", normalize=True)
    Settings.embed_model = embed_model
    Settings.persist_dir = OUTPUT_DIR
    
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
    
    index = VectorStoreIndex(nodes,   show_progress=True)
    
    # Save
    print("Saving...")

    index.storage_context.persist(OUTPUT_DIR)
    
    print(f"✅ Done! Index saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()