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
    RecursiveCharacterTextSplitter
)
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict

def setup_models():
    """ตั้งค่าโมเดลที่จำเป็น"""
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        max_length=8192,
        normalize=True
    )
    Settings.embed_model = embed_model
    return embed_model

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
        
        # 3. Recursive character splitter
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
            separators=["\n\n", "\n", " ", ""]
        )
        recursive_nodes = recursive_splitter.get_nodes_from_documents([document])
        
        # Collect results
        strategies = [
            ("Token", token_nodes),
            ("Sentence", sentence_nodes),
            ("Recursive", recursive_nodes)
        ]
        
        for strategy_name, nodes in strategies:
            if nodes:  # ตรวจสอบว่ามี nodes หรือไม่
                results.append({
                    'chunk_size': chunk_size,
                    'strategy': strategy_name,
                    'num_chunks': len(nodes),
                    'avg_chunk_length': np.mean([len(node.text) for node in nodes]),
                    'min_chunk_length': min([len(node.text) for node in nodes]),
                    'max_chunk_length': max([len(node.text) for node in nodes])
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
            print(f"Average chunk length: {np.mean([len(node.text) for node in semantic_nodes]):.2f}")
            
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
    print(f"Length consistency (std/mean): {np.std(lengths)/np.mean(lengths):.3f}")
    
    # Check for coherence (basic metric)
    coherence_scores = []
    for node in nodes:
        sentences = node.text.split('. ')
        coherence_score = len(sentences) / len(node.text)  # sentences per character
        coherence_scores.append(coherence_score)
    
    print(f"Average coherence score: {np.mean(coherence_scores):.6f}")
    
    return {
        'num_chunks': len(nodes),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'consistency': np.std(lengths)/np.mean(lengths),
        'coherence': np.mean(coherence_scores)
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
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Chunking Strategy Analysis', fontsize=16)

    # Number of chunks vs chunk size
    sns.lineplot(data=results_df, x='chunk_size', y='num_chunks', hue='strategy', ax=axes[0,0])
    axes[0,0].set_title('Number of Chunks vs Chunk Size')
    axes[0,0].set_xlabel('Chunk Size')
    axes[0,0].set_ylabel('Number of Chunks')

    # Average chunk length vs chunk size
    sns.lineplot(data=results_df, x='chunk_size', y='avg_chunk_length', hue='strategy', ax=axes[0,1])
    axes[0,1].set_title('Average Chunk Length vs Chunk Size')
    axes[0,1].set_xlabel('Chunk Size')
    axes[0,1].set_ylabel('Average Chunk Length')

    # Chunk length distribution
    strategy_colors = {'Token': 'blue', 'Sentence': 'green', 'Recursive': 'red'}
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        axes[1,0].bar(strategy_data['chunk_size'], strategy_data['max_chunk_length'], 
                     alpha=0.7, label=f'{strategy} (Max)', color=strategy_colors.get(strategy, 'gray'))

    axes[1,0].set_title('Max Chunk Length by Strategy')
    axes[1,0].set_xlabel('Chunk Size')
    axes[1,0].set_ylabel('Chunk Length')
    axes[1,0].legend()

    # Efficiency (avg chars per chunk vs num chunks)
    results_df['efficiency'] = results_df['avg_chunk_length'] / results_df['num_chunks']
    sns.barplot(data=results_df, x='chunk_size', y='efficiency', hue='strategy', ax=axes[1,1])
    axes[1,1].set_title('Efficiency (Avg Chars per Num Chunks)')
    axes[1,1].set_xlabel('Chunk Size')
    axes[1,1].set_ylabel('Efficiency')

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
    visualize_chunking_results(results_df)
    
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

if __name__ == "__main__":
    main()