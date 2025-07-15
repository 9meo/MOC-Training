#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_embedding_setup.py
Setup และ Embedding Model สำหรับ RAG Workshop

ใช้ BAAI/bge-m3 ซึ่งเป็น multilingual embedding model ที่รองรับภาษาไทย
"""

import os
import numpy as np
import torch
from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

def setup_embedding_model(model_name: str = "BAAI/bge-m3"):
    """
    ตั้งค่า BAAI/bge-m3 embedding model
    
    Args:
        model_name: ชื่อโมเดล embedding
    
    Returns:
        HuggingFaceEmbedding: โมเดล embedding ที่พร้อมใช้งาน
    """
    # Initialize BGE-M3 embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        max_length=8192,  # BGE-M3 รองรับ context length สูงสุด 8192
        normalize=True,   # normalize embeddings สำหรับ cosine similarity
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Set as global embedding model
    Settings.embed_model = embed_model
    
    print(f"Embedding model loaded: {embed_model.model_name}")
    print(f"Device: {embed_model.device}")
    print(f"Max length: {embed_model.max_length}")
    
    return embed_model

def test_embedding_model(embed_model):
    """
    ทดสอบการทำงานของ embedding model
    
    Args:
        embed_model: โมเดล embedding ที่ต้องการทดสอบ
    
    Returns:
        List: รายการ embeddings
    """
    # Test texts in Thai and English
    test_texts = [
        "สวัสดีครับ วันนี้อากาศดีมาก",
        "Hello, how are you today?",
        "การเรียนรู้ของเครื่อง (Machine Learning) เป็นเทคโนโลยีที่สำคัญ",
        "Machine Learning is an important technology"
    ]
    
    print("Testing embedding model...")
    
    # Get embeddings
    embeddings = []
    for text in test_texts:
        embedding = embed_model.get_text_embedding(text)
        embeddings.append(embedding)
        print(f"Text: {text[:50]}...")
        print(f"Embedding shape: {len(embedding)}")
        print(f"Embedding sample: {embedding[:5]}")
        print("-" * 50)
    
    # Calculate similarity between Thai and English similar texts
    thai_ml_embed = embeddings[2]
    eng_ml_embed = embeddings[3]
    
    similarity = np.dot(thai_ml_embed, eng_ml_embed) / (
        np.linalg.norm(thai_ml_embed) * np.linalg.norm(eng_ml_embed)
    )
    
    print(f"Similarity between Thai and English ML texts: {similarity:.4f}")
    
    return embeddings

def save_embedding_config(embed_model, filepath: str = "embedding_config.json"):
    """
    บันทึกการตั้งค่า embedding model
    
    Args:
        embed_model: โมเดล embedding
        filepath: ไฟล์ที่จะบันทึกข้อมูล
    """
    import json
    
    embedding_config = {
        "model_name": embed_model.model_name,
        "max_length": embed_model.max_length,
        "normalize": True,
        "device": str(embed_model.device)
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(embedding_config, f, ensure_ascii=False, indent=2)
    
    print("Embedding model configuration:")
    for key, value in embedding_config.items():
        print(f"{key}: {value}")
    
    print(f"Configuration saved to {filepath}")

if __name__ == "__main__":
    print("=" * 50)
    print("EMBEDDING MODEL SETUP")
    print("=" * 50)
    
    # Setup embedding model
    embed_model = setup_embedding_model()
    
    # Test embedding model
    test_embeddings = test_embedding_model(embed_model)
    
    # Save configuration
    save_embedding_config(embed_model)
    
    print("\nEmbedding setup completed successfully!")
    print("\nข้อดีของ BAAI/bge-m3:")
    print("1. Multilingual Support: รองรับหลายภาษารวมถึงภาษาไทย")
    print("2. High Performance: ประสิทธิภาพสูงในการ retrieval")
    print("3. Long Context: รองรับ context length สูงสุด 8,192 tokens")
    print("4. Versatile: ใช้ได้กับงานหลากหลายประเภท")