#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_indexing.py
การสร้างและจัดการ index ต่างๆ สำหรับ RAG system โดยใช้ LlamaIndex
"""

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document,
    Settings
)
from llama_index.core.indices import (
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb
import faiss
import numpy as np
import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import psutil

def setup_embedding_model():
    """ตั้งค่า embedding model"""
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        max_length=8192,
        normalize=True
    )
    Settings.embed_model = embed_model
    print(f"Embedding model loaded: {embed_model.model_name}")
    return embed_model

def create_sample_documents():
    """สร้างเอกสารตัวอย่างสำหรับการทดสอบ"""
    documents = [
        Document(
            text="""
            การเรียนรู้ของเครื่อง (Machine Learning) เป็นสาขาหนึ่งของปัญญาประดิษฐ์ที่มุ่งเน้นการพัฒนาอัลกอริทึม
            ที่สามารถเรียนรู้จากข้อมูลและปรับปรุงประสิทธิภาพได้โดยอัตโนมัติ โดยไม่ต้องเขียนโปรแกรมอย่างชัดเจน
            มีหลายประเภทหลัก ได้แก่ Supervised Learning ที่ใช้ข้อมูลที่มีป้ายกำกับ Unsupervised Learning 
            ที่ค้นหาแพทเทิร์นในข้อมูลที่ไม่มีป้ายกำกับ และ Reinforcement Learning ที่เรียนรู้จากการทดลองและรางวัล
            """,
            metadata={"source": "ml_intro.pdf", "topic": "machine_learning", "level": "beginner"}
        ),
        Document(
            text="""
            Deep Learning เป็นส่วนหนึ่งของ Machine Learning ที่ใช้โครงข่ายประสาทเทียม (Neural Networks) 
            ที่มีหลายชั้น (layers) ในการประมวลผลข้อมูล สถาปัตยกรรมที่นิยมใช้ ได้แก่ Convolutional Neural Networks (CNN) 
            สำหรับการประมวลผลภาพ Recurrent Neural Networks (RNN) สำหรับข้อมูลลำดับ และ Transformer 
            สำหรับการประมวลผลภาษาธรรมชาติ เทคโนโลยีนี้มีประสิทธิภาพสูงในการแก้ปัญหาซับซ้อนหลายประเภท
            """,
            metadata={"source": "deep_learning.pdf", "topic": "deep_learning", "level": "intermediate"}
        ),
        Document(
            text="""
            Natural Language Processing (NLP) เป็นสาขาที่รวมคอมพิวเตอร์ศาสตร์ ปัญญาประดิษฐ์ และภาษาศาสตร์
            เพื่อช่วยให้คอมพิวเตอร์เข้าใจและประมวลผลภาษาของมนุษย์ งานสำคัญใน NLP ได้แก่ การแปลภาษา 
            การสรุปข้อความ การวิเคราะห์ความรู้สึก การตอบคำถาม และการสร้างข้อความ 
            เทคโนโลยี NLP ปัจจุบันใช้ Large Language Models (LLMs) เช่น BERT, GPT, และ T5
            """,
            metadata={"source": "nlp_guide.pdf", "topic": "nlp", "level": "intermediate"}
        ),
        Document(
            text="""
            Computer Vision เป็นสาขาของปัญญาประดิษฐ์ที่มุ่งเน้นการให้คอมพิวเตอร์สามารถ"เห็น"และเข้าใจภาพ
            หรือวิดีโอได้เหมือนมนุษย์ งานหลักใน Computer Vision ได้แก่ การจำแนกภาพ (Image Classification)
            การตรวจจับวัตถุ (Object Detection) การแบ่งส่วนภาพ (Image Segmentation) และการรู้จำใบหน้า
            เทคโนโลยีนี้ถูกนำไปใช้ในรถยนต์ไร้คนขับ การวินิจฉัยทางการแพทย์ และระบบรักษาความปลอดภัย
            """,
            metadata={"source": "computer_vision.pdf", "topic": "computer_vision", "level": "advanced"}
        ),
        Document(
            text="""
            Retrieval-Augmented Generation (RAG) เป็นเทคนิคที่รวม information retrieval และ text generation
            เข้าด้วยกัน โดยการค้นหาข้อมูลที่เกี่ยวข้องจากฐานความรู้แล้วนำมาใช้ในการสร้างคำตอบ
            ขั้นตอนหลักของ RAG ประกอบด้วย การสร้าง embeddings สำหรับเอกสาร การค้นหาเอกสารที่เกี่ยวข้อง
            และการใช้ LLM สร้างคำตอบจากข้อมูลที่ค้นพบ RAG ช่วยลด hallucination และเพิ่มความแม่นยำ
            """,
            metadata={"source": "rag_tutorial.pdf", "topic": "rag", "level": "advanced"}
        )
    ]
    
    return documents

def create_simple_vector_index(documents: List[Document]):
    """
    สร้าง Simple Vector Store Index (เก็บในหน่วยความจำ)
    """
    print("Creating Simple Vector Store Index...")
    start_time = time.time()
    
    # Parse documents into nodes
    parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    
    # Create index
    index = VectorStoreIndex(nodes)
    
    end_time = time.time()
    print(f"Simple Vector Index created in {end_time - start_time:.2f} seconds")
    print(f"Number of nodes: {len(nodes)}")
    
    return index, nodes

def create_chroma_vector_index(documents: List[Document]):
    """
    สร้าง ChromaDB Vector Store Index
    """
    print("Creating ChromaDB Vector Store Index...")
    start_time = time.time()
    
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.EphemeralClient()
        chroma_collection = chroma_client.create_collection("ai_documents")
        
        # Create ChromaDB vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Parse documents
        parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(documents)
        
        # Create index
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        
        end_time = time.time()
        print(f"ChromaDB Vector Index created in {end_time - start_time:.2f} seconds")
        print(f"Number of nodes: {len(nodes)}")
        
        return index, vector_store
        
    except Exception as e:
        print(f"Error creating ChromaDB index: {str(e)}")
        print("Falling back to simple vector index...")
        return create_simple_vector_index(documents)

def create_faiss_vector_index(documents: List[Document], embed_model):
    """
    สร้าง FAISS Vector Store Index
    """
    print("Creating FAISS Vector Store Index...")
    start_time = time.time()
    
    try:
        # Parse documents
        parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(documents)
        
        # Get embedding dimension
        sample_embedding = embed_model.get_text_embedding("test")
        d = len(sample_embedding)
        
        # Create FAISS index
        faiss_index = faiss.IndexFlatIP(d)  # Inner Product index
        
        # Create FAISS vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create LlamaIndex
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        
        end_time = time.time()
        print(f"FAISS Vector Index created in {end_time - start_time:.2f} seconds")
        print(f"Number of nodes: {len(nodes)}")
        print(f"Embedding dimension: {d}")
        
        return index, vector_store
        
    except Exception as e:
        print(f"Error creating FAISS index: {str(e)}")
        print("Falling back to simple vector index...")
        return create_simple_vector_index(documents)

def create_other_indexes(documents: List[Document]):
    """
    สร้าง index ประเภทอื่นๆ
    """
    print("Creating other index types...")
    
    indexes = {}
    
    try:
        # Summary Index - สำหรับการสรุป
        summary_index = SummaryIndex.from_documents(documents)
        indexes["summary"] = summary_index
        print("Summary Index created")
    except Exception as e:
        print(f"Error creating Summary Index: {str(e)}")
    
    try:
        # Tree Index - สำหรับข้อมูลแบบลำดับชั้น
        tree_index = TreeIndex.from_documents(documents)
        indexes["tree"] = tree_index
        print("Tree Index created")
    except Exception as e:
        print(f"Error creating Tree Index: {str(e)}")
    
    try:
        # Keyword Table Index - สำหรับการค้นหาด้วยคำสำคัญ
        keyword_index = KeywordTableIndex.from_documents(documents)
        indexes["keyword"] = keyword_index
        print("Keyword Table Index created")
    except Exception as e:
        print(f"Error creating Keyword Index: {str(e)}")
    
    return indexes

def compare_index_performance(indexes: Dict[str, Any], query: str):
    """
    เปรียบเทียบประสิทธิภาพของ index ต่างๆ
    """
    results = []
    
    print(f"Testing query: {query}")
    print("=" * 50)
    
    for name, index in indexes.items():
        try:
            start_time = time.time()
            
            # Create query engine
            query_engine = index.as_query_engine(similarity_top_k=3)
            
            # Perform query
            response = query_engine.query(query)
            
            end_time = time.time()
            query_time = end_time - start_time
            
            results.append({
                "index_type": name,
                "query_time": query_time,
                "response_length": len(str(response)),
                "response_preview": str(response)[:100] + "..."
            })
            
            print(f"{name}: {query_time:.3f}s")
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            results.append({
                "index_type": name,
                "query_time": -1,
                "response_length": 0,
                "response_preview": f"Error: {str(e)}"
            })
    
    return results

def create_advanced_vector_index(documents: List[Document]):
    """
    สร้าง Vector Index ที่มีการตั้งค่าขั้นสูง
    """
    print("Creating Advanced Vector Index...")
    
    # Advanced node parser with metadata
    parser = SentenceSplitter(
        chunk_size=400,
        chunk_overlap=20,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
    )
    
    # Parse documents
    nodes = parser.get_nodes_from_documents(documents)
    
    # Add additional metadata to nodes
    for i, node in enumerate(nodes):
        node.metadata.update({
            "node_id": f"advanced_node_{i}",
            "chunk_size": len(node.text),
            "word_count": len(node.text.split()),
            "index_type": "advanced_vector"
        })
    
    # Create index with custom settings
    index = VectorStoreIndex(
        nodes,
        show_progress=True,
        use_async=False
    )
    
    print(f"Advanced Vector Index created with {len(nodes)} nodes")
    
    return index, nodes

def save_and_load_index(index, index_name: str):
    """
    บันทึกและโหลด index
    """
    storage_dir = f"./storage/{index_name}"
    os.makedirs(storage_dir, exist_ok=True)
    
    try:
        # Save index
        print(f"Saving {index_name} index...")
        index.storage_context.persist(persist_dir=storage_dir)
        print(f"Index saved to {storage_dir}")
        
        # Load index
        print(f"Loading {index_name} index...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        loaded_index = load_index_from_storage(storage_context)
        print(f"Index loaded successfully")
        
        return loaded_index
        
    except Exception as e:
        print(f"Error saving/loading index: {str(e)}")
        return None

def analyze_index_statistics(nodes: List, index_name: str):
    """
    วิเคราะห์สถิติของ index
    """
    print(f"\n{index_name} Index Statistics:")
    print("=" * 40)
    
    if not nodes:
        print("No nodes available for analysis")
        return {}
    
    # Basic statistics
    total_nodes = len(nodes)
    total_chars = sum(len(node.text) for node in nodes)
    avg_chars = total_chars / total_nodes if total_nodes > 0 else 0
    
    print(f"Total nodes: {total_nodes}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average characters per node: {avg_chars:.2f}")
    
    # Node size distribution
    node_sizes = [len(node.text) for node in nodes]
    print(f"Min node size: {min(node_sizes)} characters")
    print(f"Max node size: {max(node_sizes)} characters")
    print(f"Median node size: {np.median(node_sizes):.2f} characters")
    
    # Metadata analysis
    if nodes and hasattr(nodes[0], 'metadata'):
        topics = [node.metadata.get('topic', 'unknown') for node in nodes]
        unique_topics = set(topics)
        print(f"Unique topics: {len(unique_topics)}")
        print(f"Topics: {', '.join(unique_topics)}")
        
        # Topic distribution
        topic_counts = {topic: topics.count(topic) for topic in unique_topics}
        print("Topic distribution:")
        for topic, count in topic_counts.items():
            print(f"  {topic}: {count} nodes")
    
    return {
        "total_nodes": total_nodes,
        "total_chars": total_chars,
        "avg_chars": avg_chars,
        "min_size": min(node_sizes),
        "max_size": max(node_sizes),
        "median_size": np.median(node_sizes)
    }

def analyze_memory_usage():
    """
    วิเคราะห์การใช้หน่วยความจำของ indexes
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        print("\nMemory Usage Analysis:")
        print("=" * 30)
        print(f"RSS (Resident Set Size): {memory_info.rss / 1024 / 1024:.2f} MB")
        print(f"VMS (Virtual Memory Size): {memory_info.vms / 1024 / 1024:.2f} MB")
        print(f"Memory Percent: {process.memory_percent():.2f}%")
    except:
        print("Memory analysis not available")

def visualize_performance(results):
    """
    แสดงผลการเปรียบเทียบประสิทธิภาพ
    """
    df = pd.DataFrame(results)
    
    # Filter out error results
    df_valid = df[df['query_time'] > 0]
    
    if len(df_valid) > 0:
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Query time comparison
        sns.barplot(data=df_valid, x='index_type', y='query_time', ax=axes[0])
        axes[0].set_title('Query Time Comparison')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_xlabel('Index Type')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Response length comparison
        sns.barplot(data=df_valid, x='index_type', y='response_length', ax=axes[1])
        axes[1].set_title('Response Length Comparison')
        axes[1].set_ylabel('Response Length (characters)')
        axes[1].set_xlabel('Index Type')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    # Print detailed results
    print("\nDetailed Performance Results:")
    print(df.to_string(index=False))
    
    return df

def export_index_info(simple_stats, advanced_stats, performance_results):
    """
    ส่งออกข้อมูล index สำหรับใช้ในขั้นตอนต่อไป
    """
    index_info = {
        "indexes": {
            "simple_vector": {
                "type": "VectorStoreIndex",
                "storage": "memory",
                "embedding_model": "BAAI/bge-m3"
            },
            "chroma_vector": {
                "type": "ChromaVectorStore",
                "storage": "chromadb",
                "embedding_model": "BAAI/bge-m3"
            },
            "faiss_vector": {
                "type": "FaissVectorStore",
                "storage": "faiss",
                "embedding_model": "BAAI/bge-m3"
            }
        },
        "statistics": {
            "simple": simple_stats,
            "advanced": advanced_stats
        },
        "performance": performance_results
    }
    
    # Save to JSON
    with open('index_info.json', 'w', encoding='utf-8') as f:
        json.dump(index_info, f, ensure_ascii=False, indent=2)
    
    print("Index information exported to index_info.json")
    return index_info

def main():
    """ฟังก์ชันหลักสำหรับทดสอบ indexing"""
    print("=" * 60)
    print("INDEXING WITH LLAMAINDEX")
    print("=" * 60)
    
    # Setup embedding model
    embed_model = setup_embedding_model()
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")
    
    # 1. Create Simple Vector Store Index
    simple_index, nodes = create_simple_vector_index(documents)
    
    # 2. Create ChromaDB Vector Store Index
    chroma_index, chroma_store = create_chroma_vector_index(documents)
    
    # 3. Create FAISS Vector Store Index
    faiss_index, faiss_store = create_faiss_vector_index(documents, embed_model)
    
    # 4. Create other index types
    other_indexes = create_other_indexes(documents)
    
    # 5. Create advanced vector index
    advanced_index, advanced_nodes = create_advanced_vector_index(documents)
    
    # 6. Performance comparison
    test_query = "Machine Learning คืออะไร และมีประเภทใดบ้าง?"
    
    # Prepare indexes for testing
    test_indexes = {
        "Simple Vector": simple_index,
        "Advanced Vector": advanced_index
    }
    
    # Add other indexes if they were created successfully
    for name, index in other_indexes.items():
        test_indexes[f"{name.title()} Index"] = index
    
    # Add chroma and faiss if available
    if 'chroma_index' in locals():
        test_indexes["ChromaDB"] = chroma_index
    if 'faiss_index' in locals():
        test_indexes["FAISS"] = faiss_index
    
    performance_results = compare_index_performance(test_indexes, test_query)
    
    # 7. Visualize performance
    performance_df = visualize_performance(performance_results)
    
    # 8. Analyze statistics
    simple_stats = analyze_index_statistics(nodes, "Simple Vector")
    advanced_stats = analyze_index_statistics(advanced_nodes, "Advanced Vector")
    
    # 9. Memory analysis
    analyze_memory_usage()
    
    # 10. Save and load test
    loaded_simple_index = save_and_load_index(simple_index, "simple_vector")
    
    # 11. Export index information
    index_info = export_index_info(simple_stats, advanced_stats, performance_results)
    
    print("\n" + "=" * 60)
    print("สรุป Indexing Strategies")
    print("=" * 60)
    print("\nประเภท Index ที่ใช้:")
    print("1. VectorStoreIndex: เหมาะสำหรับ semantic search")
    print("2. ChromaDB: เหมาะสำหรับการเก็บข้อมูลขนาดใหญ่")
    print("3. FAISS: เหมาะสำหรับการค้นหาที่รวดเร็ว")
    print("4. SummaryIndex: เหมาะสำหรับการสรุปเนื้อหา")
    print("5. KeywordTableIndex: เหมาะสำหรับการค้นหาด้วยคำสำคัญ")
    print("\nแนะนำการใช้งาน:")
    print("- ข้อมูลขนาดเล็ก (< 1,000 documents): ใช้ VectorStoreIndex")
    print("- ข้อมูลขนาดกลาง (1,000-10,000 documents): ใช้ ChromaDB")
    print("- ข้อมูลขนาดใหญ่ (> 10,000 documents): ใช้ FAISS")
    print("- การสรุป: ใช้ SummaryIndex ร่วมกับ VectorStoreIndex")
    print("- การค้นหาแบบผสม: รวม VectorStore และ KeywordTable")
    print("\nข้อควรพิจารณา:")
    print("- Memory Usage: Vector indexes ใช้หน่วยความจำมาก")
    print("- Query Speed: FAISS เร็วที่สุด, ChromaDB สมดุล")
    print("- Persistence: ChromaDB และ FAISS รองรับการบันทึกข้อมูล")
    print("- Scalability: FAISS เหมาะสำหรับการขยายขนาด")
    
    return {
        "simple_index": simple_index,
        "advanced_index": advanced_index,
        "other_indexes": other_indexes,
        "performance_results": performance_results,
        "nodes": nodes,
        "advanced_nodes": advanced_nodes
    }

if __name__ == "__main__":
    results = main()