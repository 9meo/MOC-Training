#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

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
    docs = SimpleDirectoryReader(input_dir=DATA_DIR).load_data()
    # ตั้งค่าให้แยกเอกสารเป็น chunk ย่อย ๆ
    text_splitter = SentenceSplitter(
        separator=" ",       # ใช้เว้นวรรคเป็นตัวแยก
        chunk_size=128,     # ขนาดของแต่ละ chunk (จำนวนตัวอักษร)
        chunk_overlap=10    # ความซ้อนของข้อความระหว่างแต่ละ chunk
    )

    # สร้าง ingestion pipeline ที่ประกอบด้วย text_splitter
    pipeline = IngestionPipeline(
        transformations=[
            text_splitter,
        ]
    )

    # รัน pipeline กับเอกสารที่โหลดไว้ เพื่อแยกเป็น chunks
    nodes = pipeline.run(
        documents=docs,
        in_place=True,       # แก้ไขเอกสารเดิมแทนที่จะสร้างใหม่
        show_progress=True,  # แสดง progress bar ขณะรัน
    )
    print(f"Created {len(nodes)} chunks from {len(docs)} documents")
    
    # เริ่มต้น client ของ ChromaDB และสร้าง collection สำหรับเก็บเวกเตอร์
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("rag_demo")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # สร้าง storage context สำหรับบันทึกข้อมูล
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # สร้าง index จากเอกสาร โดยใช้ embedding ที่เรากำหนด
    index = VectorStoreIndex(
        nodes=nodes, storage_context=storage_context, embed_model=embed_model,show_progress=True,  # แสดง progress bar ขณะรัน
    )
    
    # Save
    print("Saving...")

    index.storage_context.persist(OUTPUT_DIR)
    
    print(f"✅ Done! Index saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()