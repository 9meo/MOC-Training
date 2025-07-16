#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_chunking_strategies.py
กลยุทธ์การแบ่ง chunks และผลกระทบของขนาด chunk ต่อประสิทธิภาพ
"""

from llama_index.core.node_parser import (
    TokenTextSplitter,
    SentenceSplitter,
    SemanticSplitterNodeParser,
    SimpleNodeParser
)
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from typing import List, Dict
import re

def setup_models():
    """ตั้งค่าโมเดลที่จำเป็น"""
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        max_length=8192,
        normalize=True,
        device=device
    )
    Settings.embed_model = embed_model
    print(f"Embedding model loaded on: {device}")
    return embed_model

class CustomRecursiveTextSplitter:
    """
    Custom implementation of recursive text splitter for LlamaIndex
    """
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def get_nodes_from_documents(self, documents: List[Document]) -> List:
        """Split documents into nodes using recursive approach"""
        nodes = []
        for doc in documents:
            text_chunks = self._split_text_recursive(doc.text, self.separators)
            for chunk in text_chunks:
                if chunk.strip():
                    nodes.append(type('Node', (), {'text': chunk.strip()})())
        return nodes
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using different separators"""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Character-level splitting
            return self._split_by_characters(text)
        
        chunks = []
        parts = text.split(separator)
        
        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(part) > self.chunk_size:
                    # Part is too long, split it further
                    sub_chunks = self._split_text_recursive(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_characters(self, text: str) -> List[str]:
        """Split text by characters when all other separators fail"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

def create_sample_document():
    """สร้างเอกสารตัวอย่างภาษาไทยสำหรับการทดสอบ"""
    sample_thai_text = """
    ปัญญาประดิษฐ์ (Artificial Intelligence) เป็นเทคโนโลยีที่กำลังเปลี่ยนแปลงโลกในยุคปัจจุบัน การพัฒนาด้านปัญญาประดิษฐ์ได้ก้าวหน้าอย่างรวดเร็ว โดยเฉพาะในด้านการเรียนรู้ของเครื่อง (Machine Learning) และการเรียนรู้เชิงลึก (Deep Learning)

    การเรียนรู้ของเครื่องเป็นสาขาหนึ่งของปัญญาประดิษฐ์ที่มุ่งเน้นการพัฒนาอัลกอริทึมที่สามารถเรียนรู้จากข้อมูลและทำนายผลลัพธ์ได้ โดยไม่ต้องเขียนโปรแกรมอย่างชัดเจน มีหลายประเภทของการเรียนรู้ของเครื่อง เช่น Supervised Learning, Unsupervised Learning, และ Reinforcement Learning

    การเรียนรู้เชิงลึกเป็นส่วนหนึ่งของการเรียนรู้ของเครื่อง ที่ใช้โครงข่ายประสาทเทียม (Neural Networks) ที่มีหลายชั้น (layers) ในการประมวลผลข้อมูล เทคโนโลยีนี้ได้รับความนิยมอย่างมากในการแก้ปัญหาที่ซับซ้อน เช่น การรู้จำภาพ การประมวลผลภาษาธรรมชาติ และการสร้างเนื้อหา

    ในปัจจุบัน Large Language Models (LLMs) เช่น GPT, BERT, และ T5 ได้กลายเป็นเครื่องมือสำคัญในการประมวลผลภาษาธรรมชาติ โมเดลเหล่านี้ได้รับการฝึกฝนด้วยข้อมูลขนาดใหญ่และสามารถทำงานหลากหลายได้ เช่น การแปลภาษา การสรุปข้อความ การตอบคำถาม และการเขียนเนื้อหา

    การประยุกต์ใช้ปัญญาประดิษฐ์ในอุตสาหกรรมต่างๆ ได้แก่ อุตสาหกรรมยานยนต์ (รถยนต์ไร้คนขับ) อุตสาหกรรมการเงิน (การวิเคราะห์ความเสี่ยง) อุตสาหกรรมสุขภาพ (การวินิจฉัยโรค) และอุตสาหกรรมเกษตร (การจัดการฟาร์มอัจฉริยะ)

    อย่างไรก็ตาม การพัฒนาปัญญาประดิษฐ์ยังมีความท้าทายหลายประการ เช่น ปัญหาด้านจริยธรรม ความเป็นส่วนตัว ความปลอดภัย และผลกระทบต่อตลาดแรงงาน สิ่งเหล่านี้ต้องได้รับการพิจารณาอย่างรอบคอบในการพัฒนาเทคโนโลยีในอนาคต
    """
    
    document = Document(text=sample_thai_text.strip())
    print(f"Document length: {len(document.text)} characters")
    print(f"Document preview: {document.text[:200]}...")
    
    return document

def compare_chunking_strategies(document: Document, chunk_sizes: List[int]):
    """
    เปรียบเทียบกลยุทธ์การแบ่ง chunk ต่างๆ
    
    Args:
        document: เอกสารที่ต้องการแบ่ง chunk
        chunk_sizes: รายการขนาด chunk ที่ต้องการทดสอบ
    
    Returns:
        DataFrame: ผลการเปรียบเทียบ
    """
    results = []
    
    for chunk_size in chunk_sizes:
        print(f"Testing chunk size: {chunk_size}")
        
        # 1. Token-based splitter
        token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1)  # 10% overlap
        )
        token_nodes = token_splitter.get_nodes_from_documents([document])
        
        # 2. Sentence-based splitter
        sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1)
        )
        sentence_nodes = sentence_splitter.get_nodes_from_documents([document])
        
        # 3. Custom recursive character splitter
        recursive_splitter = CustomRecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
            separators=["\n\n", "\n", " ", ""]
        )
        recursive_nodes = recursive_splitter.get_nodes_from_documents([document])
        
        # 4. Simple node parser (basic splitting)
        simple_parser = SimpleNodeParser(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1)
        )
        simple_nodes = simple_parser.get_nodes_from_documents([document])
        
        # Collect results
        strategies = [
            ("Token", token_nodes),
            ("Sentence", sentence_nodes),
            ("Recursive", recursive_nodes),
            ("Simple", simple_nodes)
        ]
        
        for strategy_name, nodes in strategies:
            if nodes:  # ตรวจสอบว่ามี nodes หรือไม่
                chunk_lengths = [len(node.text) for node in nodes]
                results.append({
                    'chunk_size': chunk_size,
                    'strategy': strategy_name,
                    'num_chunks': len(nodes),
                    'avg_chunk_length': np.mean(chunk_lengths),
                    'min_chunk_length': min(chunk_lengths),
                    'max_chunk_length': max(chunk_lengths),
                    'std_chunk_length': np.std(chunk_lengths)
                })
    
    return pd.DataFrame(results)

def test_semantic_chunking(document: Document, embed_model):
    """
    ทดสอบ semantic chunking ที่ใช้ embedding เพื่อแบ่ง chunk ตามความหมาย
    
    Args:
        document: เอกสารที่ต้องการแบ่ง
        embed_model: โมเดล embedding
    
    Returns:
        List: รายการ semantic nodes
    """
    print("Testing Semantic Chunking...")
    
    try:
        # Create semantic splitter
        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,  # Number of sentences to group together
            breakpoint_percentile_threshold=95,  # Threshold for semantic breaks
            embed_model=embed_model
        )
        
        # Get semantic nodes
        semantic_nodes = semantic_splitter.get_nodes_from_documents([document])
        
        print(f"Semantic Chunking Results:")
        print(f"Number of semantic chunks: {len(semantic_nodes)}")
        if semantic_nodes:
            chunk_lengths = [len(node.text) for node in semantic_nodes]
            print(f"Average chunk length: {np.mean(chunk_lengths):.2f}")
            print(f"Min chunk length: {min(chunk_lengths)}")
            print(f"Max chunk length: {max(chunk_lengths)}")
            
            # Show semantic chunks
            for i, node in enumerate(semantic_nodes[:3]):  # แสดงแค่ 3 chunks แรก
                print(f"\n--- Semantic Chunk {i+1} ---")
                print(f"Length: {len(node.text)} characters")
                print(f"Content: {node.text[:200]}...")
        
        return semantic_nodes
        
    except Exception as e:
        print(f"Error in semantic chunking: {str(e)}")
        return []

def analyze_chunk_quality(nodes: List, chunk_name: str):
    """
    วิเคราะห์คุณภาพของ chunks
    
    Args:
        nodes: รายการ nodes
        chunk_name: ชื่อประเภท chunk
    
    Returns:
        Dict: สถิติคุณภาพ
    """
    if not nodes:
        print(f"No nodes available for {chunk_name}")
        return {}
    
    lengths = [len(node.text) for node in nodes]
    
    print(f"\n{chunk_name} Quality Analysis:")
    print(f"Number of chunks: {len(nodes)}")
    print(f"Average length: {np.mean(lengths):.2f}")
    print(f"Standard deviation: {np.std(lengths):.2f}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    
    if np.mean(lengths) > 0:
        print(f"Length consistency (std/mean): {np.std(lengths)/np.mean(lengths):.3f}")
    
    # Check for coherence (basic metric)
    coherence_scores = []
    for node in nodes:
        sentences = node.text.split('. ')
        if len(node.text) > 0:
            coherence_score = len(sentences) / len(node.text)  # sentences per character
            coherence_scores.append(coherence_score)
    
    if coherence_scores:
        print(f"Average coherence score: {np.mean(coherence_scores):.6f}")
    
    return {
        'num_chunks': len(nodes),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'consistency': np.std(lengths)/np.mean(lengths) if np.mean(lengths) > 0 else 0,
        'coherence': np.mean(coherence_scores) if coherence_scores else 0
    }

def visualize_chunking_results(results_df: pd.DataFrame):
    """
    แสดงผลการเปรียบเทียบการแบ่ง chunk
    
    Args:
        results_df: DataFrame ผลการเปรียบเทียบ
    """
    if results_df.empty:
        print("No results to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Chunking Strategy Analysis', fontsize=16, fontweight='bold')

    # Number of chunks vs chunk size
    sns.lineplot(data=results_df, x='chunk_size', y='num_chunks', hue='strategy', 
                marker='o', linewidth=2, markersize=8, ax=axes[0,0])
    axes[0,0].set_title('Number of Chunks vs Chunk Size', fontsize=14)
    axes[0,0].set_xlabel('Chunk Size', fontsize=12)
    axes[0,0].set_ylabel('Number of Chunks', fontsize=12)
    axes[0,0].grid(True, alpha=0.3)

    # Average chunk length vs chunk size
    sns.lineplot(data=results_df, x='chunk_size', y='avg_chunk_length', hue='strategy', 
                marker='s', linewidth=2, markersize=8, ax=axes[0,1])
    axes[0,1].set_title('Average Chunk Length vs Chunk Size', fontsize=14)
    axes[0,1].set_xlabel('Chunk Size', fontsize=12)
    axes[0,1].set_ylabel('Average Chunk Length', fontsize=12)
    axes[0,1].grid(True, alpha=0.3)

    # Chunk length standard deviation
    sns.lineplot(data=results_df, x='chunk_size', y='std_chunk_length', hue='strategy', 
                marker='^', linewidth=2, markersize=8, ax=axes[1,0])
    axes[1,0].set_title('Chunk Length Standard Deviation', fontsize=14)
    axes[1,0].set_xlabel('Chunk Size', fontsize=12)
    axes[1,0].set_ylabel('Standard Deviation', fontsize=12)
    axes[1,0].grid(True, alpha=0.3)

    # Efficiency (avg chars per chunk vs num chunks)
    results_df['efficiency'] = results_df['avg_chunk_length'] / results_df['num_chunks']
    sns.lineplot(data=results_df, x='chunk_size', y='efficiency', hue='strategy', 
                marker='d', linewidth=2, markersize=8, ax=axes[1,1])
    axes[1,1].set_title('Efficiency (Avg Length / Num Chunks)', fontsize=14)
    axes[1,1].set_xlabel('Chunk Size', fontsize=12)
    axes[1,1].set_ylabel('Efficiency', fontsize=12)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    """ฟังก์ชันหลักสำหรับทดสอบ chunking strategies"""
    print("=" * 60)
    print("CHUNKING STRATEGIES ANALYSIS")
    print("=" * 60)
    
    # Setup models
    embed_model = setup_models()
    
    # Create sample document
    document = create_sample_document()
    
    # Test different chunk sizes
    chunk_sizes = [100, 200, 500, 1000, 2000]
    print(f"\nTesting chunk sizes: {chunk_sizes}")
    
    # Compare chunking strategies
    results_df = compare_chunking_strategies(document, chunk_sizes)
    
    print("\nChunking Strategy Comparison:")
    print(results_df.to_string(index=False))
    
    # Visualize results
    print("\nGenerating visualizations...")
    try:
        visualize_chunking_results(results_df)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        print("Continuing without visualizations...")
    
    # Test semantic chunking
    semantic_nodes = test_semantic_chunking(document, embed_model)
    
    # Analyze chunk quality for different approaches
    print("\n" + "=" * 40)
    print("CHUNK QUALITY ANALYSIS")
    print("=" * 40)
    
    # Compare different chunking approaches
    sentence_splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    sentence_nodes = sentence_splitter.get_nodes_from_documents([document])
    
    token_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    token_nodes = token_splitter.get_nodes_from_documents([document])
    
    # Analyze quality
    sentence_quality = analyze_chunk_quality(sentence_nodes, "Sentence-based")
    token_quality = analyze_chunk_quality(token_nodes, "Token-based")
    if semantic_nodes:
        semantic_quality = analyze_chunk_quality(semantic_nodes, "Semantic-based")
    
    # Summary and recommendations
    print("\n" + "=" * 40)
    print("สรุปและคำแนะนำ")
    print("=" * 40)
    print("\nผลกระทบของขนาด Chunk:")
    print("\n• ขนาด Chunk เล็ก (100-200 tokens)")
    print("  - ข้อดี: ความแม่นยำสูง, ค้นหาข้อมูลเฉพาะเจาะจง")
    print("  - ข้อเสีย: สูญเสีย context, จำนวน chunk มาก")
    print("\n• ขนาด Chunk กลาง (500-1000 tokens)")
    print("  - ข้อดี: สมดุลระหว่างความแม่นยำและ context")
    print("  - ข้อเสีย: อาจมีข้อมูลที่ไม่เกี่ยวข้องปนมา")
    print("\n• ขนาด Chunk ใหญ่ (1000+ tokens)")
    print("  - ข้อดี: context ครบถ้วน, จำนวน chunk น้อย")
    print("  - ข้อเสีย: ความแม่นยำลดลง, เสียเวลาในการประมวลผล")
    print("\nคำแนะนำ:")
    print("• ภาษาไทย: ใช้ขนาด 300-800 tokens เนื่องจากความซับซ้อนของภาษา")
    print("• เอกสารเทคนิค: ใช้ semantic chunking")
    print("• เนื้อหาทั่วไป: ใช้ sentence-based chunking")
    print("• การทดสอบ: ทดสอบขนาด chunk ต่างๆ เพื่อหาค่าที่เหมาะสม")

if __name__ == "__main__":
    main()