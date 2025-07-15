#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
searching_and_querying_05.py
การค้นหาและสร้างคำตอบด้วย RAG system โดยใช้ query engines และ retrievers ต่างๆ
"""

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import time
import pandas as pd
from typing import List, Dict

def setup_models():
    """ตั้งค่าโมเดลที่จำเป็น"""
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        max_length=8192,
        normalize=True
    )
    Settings.embed_model = embed_model
    print("✅ Models loaded successfully")
    return embed_model

def create_documents():
    """สร้างเอกสารตัวอย่าง"""
    documents = [
        Document(
            text="""
            การเรียนรู้ของเครื่อง (Machine Learning) เป็นสาขาหนึ่งของปัญญาประดิษฐ์ที่ให้คอมพิวเตอร์สามารถเรียนรู้และปรับปรุงประสิทธิภาพ
            จากประสบการณ์โดยไม่ต้องเขียนโปรแกรมอย่างชัดเจน มีสามประเภทหลัก คือ Supervised Learning ที่ใช้ข้อมูลที่มีป้ายกำกับ
            Unsupervised Learning ที่ค้นหาแพทเทิร์นในข้อมูลที่ไม่มีป้ายกำกับ และ Reinforcement Learning ที่เรียนรู้จากการโต้ตอบกับสภาพแวดล้อม
            อัลกอริทึมที่นิยมใช้ ได้แก่ Linear Regression, Decision Trees, Random Forest, Support Vector Machines และ Neural Networks
            """,
            metadata={"topic": "machine_learning", "difficulty": "beginner"}
        ),
        Document(
            text="""
            Deep Learning เป็นส่วนหนึ่งของ Machine Learning ที่ใช้โครงข่ายประสาทเทียมที่มีหลายชั้น (Deep Neural Networks)
            สถาปัตยกรรมที่สำคัญ ได้แก่ CNN สำหรับการประมวลผลภาพ RNN และ LSTM สำหรับข้อมูลลำดับ และ Transformer 
            สำหรับการประมวลผลภาษาธรรมชาติ การฝึกฝน Deep Learning ต้องการข้อมูลจำนวนมากและทรัพยากรการคำนวณสูง
            """,
            metadata={"topic": "deep_learning", "difficulty": "intermediate"}
        ),
        Document(
            text="""
            Natural Language Processing (NLP) เป็นสาขาที่รวมคอมพิวเตอร์ศาสตร์ ปัญญาประดิษฐ์ และภาษาศาสตร์
            เพื่อให้คอมพิวเตอร์เข้าใจและประมวลผลภาษาของมนุษย์ งานหลักใน NLP ประกอบด้วย Text Classification
            Named Entity Recognition (NER) Sentiment Analysis Machine Translation Question Answering และ Text Summarization
            เทคนิคสมัยใหม่ใช้ Pre-trained Language Models เช่น BERT GPT และ T5 ที่ผ่านการฝึกฝนด้วยข้อมูลขนาดใหญ่
            """,
            metadata={"topic": "nlp", "difficulty": "intermediate"}
        ),
        Document(
            text="""
            Retrieval-Augmented Generation (RAG) เป็นเทคนิคที่รวม Information Retrieval และ Text Generation เข้าด้วยกัน
            ประกอบด้วยสองส่วนหลัก คือ Retriever ที่ค้นหาเอกสารที่เกี่ยวข้องจากฐานความรู้ และ Generator ที่สร้างคำตอบ
            จากข้อมูลที่ค้นพบ ขั้นตอนการทำงานเริ่มจากการสร้าง embeddings สำหรับเอกสาร จากนั้นเมื่อมีคำถาม
            ระบบจะค้นหาเอกสารที่มี similarity สูงสุด แล้วนำมาใช้เป็น context ในการสร้างคำตอบด้วย Language Model
            RAG ช่วยลด hallucination และเพิ่มความแม่นยำของ AI ในการตอบคำถาม
            """,
            metadata={"topic": "rag", "difficulty": "advanced"}
        )
    ]
    return documents

def create_index(documents):
    """สร้าง index จากเอกสาร"""
    parser = SentenceSplitter(chunk_size=400, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    print(f"📚 Created index with {len(nodes)} nodes from {len(documents)} documents")
    return index, nodes

def test_basic_query_engine(index, queries):
    """ทดสอบ Basic Query Engine"""
    print("\n🔍 Testing Basic Query Engine:")
    print("=" * 40)
    
    query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = query_engine.query(query)
            end_time = time.time()
            
            result = {
                "query": query,
                "response": str(response),
                "response_time": end_time - start_time,
                "source_nodes": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
            }
            results.append(result)
            
            print(f"Answer: {str(response)[:200]}...")
            print(f"Time: {end_time - start_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "query": query,
                "response": f"Error: {str(e)}",
                "response_time": 0,
                "source_nodes": 0
            })
    
    return results

def test_response_modes(index, query):
    """ทดสอบ Response Modes ต่างๆ"""
    print(f"\n📝 Testing Response Modes with: {query}")
    print("=" * 50)
    
    modes = ["compact", "refine", "tree_summarize", "simple_summarize"]
    results = []
    
    for mode in modes:
        try:
            print(f"\n🔄 Testing {mode} mode:")
            
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                response_mode=mode
            )
            
            start_time = time.time()
            response = query_engine.query(query)
            end_time = time.time()
            
            result = {
                "mode": mode,
                "response": str(response),
                "response_time": end_time - start_time,
                "response_length": len(str(response))
            }
            results.append(result)
            
            print(f"Response ({len(str(response))} chars): {str(response)[:150]}...")
            print(f"Time: {end_time - start_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Error with {mode}: {str(e)}")
            results.append({
                "mode": mode,
                "response": f"Error: {str(e)}",
                "response_time": -1,
                "response_length": 0
            })
    
    return results

def test_advanced_query_engine(index, queries):
    """ทดสอบ Advanced Query Engine"""
    print("\n⚡ Testing Advanced Query Engine:")
    print("=" * 40)
    
    try:
        # Create custom retriever
        retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        
        # Add postprocessors
        postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.6),
            KeywordNodePostprocessor(keywords=["machine learning", "deep learning"], exclude_keywords=[])
        ]
        
        # Create advanced query engine
        query_engine = index.as_query_engine(similarity_top_k=5)  # Fallback to basic
        
    except Exception as e:
        print(f"⚠️ Advanced features not available: {str(e)}")
        print("Using basic query engine instead...")
        query_engine = index.as_query_engine(similarity_top_k=5)
    
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\nAdvanced Query {i}: {query}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = query_engine.query(query)
            end_time = time.time()
            
            source_info = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_info.append({
                        "score": getattr(node, 'score', 0),
                        "topic": node.metadata.get('topic', 'unknown'),
                        "text_preview": node.text[:100] + "..."
                    })
            
            result = {
                "query": query,
                "response": str(response),
                "response_time": end_time - start_time,
                "sources": source_info
            }
            results.append(result)
            
            print(f"Answer: {str(response)[:200]}...")
            print(f"Sources: {len(source_info)} nodes")
            print(f"Time: {end_time - start_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "query": query,
                "response": f"Error: {str(e)}",
                "response_time": 0,
                "sources": []
            })
    
    return results

def compare_configurations(index, queries):
    """เปรียบเทียบการตั้งค่าต่างๆ"""
    print("\n📊 Comparing Query Engine Configurations:")
    print("=" * 50)
    
    configs = [
        {"name": "Basic", "similarity_top_k": 3, "response_mode": "compact"},
        {"name": "High Recall", "similarity_top_k": 7, "response_mode": "tree_summarize"},
        {"name": "Precise", "similarity_top_k": 2, "response_mode": "compact"},
        {"name": "Fast", "similarity_top_k": 3, "response_mode": "simple_summarize"}
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n🔧 Testing {config['name']} configuration:")
        
        for query in queries[:2]:  # Test with first 2 queries only
            try:
                query_engine = index.as_query_engine(
                    similarity_top_k=config['similarity_top_k'],
                    response_mode=config['response_mode']
                )
                
                start_time = time.time()
                response = query_engine.query(query)
                end_time = time.time()
                
                all_results.append({
                    "config": config['name'],
                    "query": query[:50] + "...",
                    "response_time": end_time - start_time,
                    "response_length": len(str(response)),
                    "similarity_top_k": config['similarity_top_k'],
                    "response_mode": config['response_mode']
                })
                
                print(f"  Query: {query[:50]}...")
                print(f"  Time: {end_time - start_time:.3f}s, Length: {len(str(response))} chars")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
    
    return all_results

def analyze_performance(results):
    """วิเคราะห์ประสิทธิภาพ"""
    if not results:
        print("No results to analyze")
        return
    
    df = pd.DataFrame(results)
    
    print("\n📈 Performance Analysis:")
    print("=" * 30)
    
    if 'config' in df.columns:
        # Group by configuration
        summary = df.groupby('config').agg({
            'response_time': ['mean', 'std'],
            'response_length': ['mean', 'std']
        }).round(3)
        
        print("Performance Summary by Configuration:")
        print(summary)
    else:
        # Overall statistics
        print(f"Total queries: {len(df)}")
        print(f"Average response time: {df['response_time'].mean():.3f}s")
        print(f"Average response length: {df['response_length'].mean():.0f} chars")

def main():
    """ฟังก์ชันหลักสำหรับ demo การค้นหาและ querying"""
    print("=" * 60)
    print("🔍 SEARCHING AND QUERYING WITH RAG")
    print("=" * 60)
    
    try:
        # 1. Setup
        print("\n1️⃣ Setting up models...")
        embed_model = setup_models()
        
        # 2. Create documents and index
        print("\n2️⃣ Creating documents and index...")
        documents = create_documents()
        index, nodes = create_index(documents)
        
        # 3. Test queries
        test_queries = [
            "Machine Learning คืออะไร และมีประเภทใดบ้าง?",
            "ความแตกต่างระหว่าง CNN และ RNN คืออะไร?",
            "RAG ทำงานอย่างไร?",
            "NLP ใช้เทคนิคอะไรบ้าง?"
        ]
        
        # 4. Basic query engine test
        print("\n3️⃣ Testing Basic Query Engine...")
        basic_results = test_basic_query_engine(index, test_queries)
        
        # 5. Response modes test
        print("\n4️⃣ Testing Response Modes...")
        mode_results = test_response_modes(
            index, 
            "อธิบายความแตกต่างระหว่าง Machine Learning และ Deep Learning"
        )
        
        # 6. Advanced query engine test
        print("\n5️⃣ Testing Advanced Query Engine...")
        advanced_results = test_advanced_query_engine(index, test_queries[:2])
        
        # 7. Configuration comparison
        print("\n6️⃣ Comparing Configurations...")
        comparison_results = compare_configurations(index, test_queries)
        
        # 8. Performance analysis
        print("\n7️⃣ Performance Analysis...")
        analyze_performance(basic_results)
        if comparison_results:
            analyze_performance(comparison_results)
        
        # 9. Summary
        print("\n" + "=" * 60)
        print("📋 SUMMARY - Searching and Querying Strategies")
        print("=" * 60)
        print("\n✅ Query Engine Types:")
        print("1. Basic Query Engine: เหมาะสำหรับการใช้งานทั่วไป")
        print("2. Advanced Query Engine: ใช้ postprocessors และ filters")
        print("\n✅ Response Modes:")
        print("- compact: รวม context ทั้งหมด (เร็ว)")
        print("- refine: ปรับปรุงคำตอบแบบ iterative (แม่นยำ)")
        print("- tree_summarize: สรุปแบบต้นไม้ (สมดุล)")
        print("- simple_summarize: สรุปแบบง่าย (เร็ว)")
        print("\n✅ Best Practices:")
        print("- เลือก similarity_top_k ที่เหมาะสม (3-8 nodes)")
        print("- ปรับ response_mode ตามความต้องการ")
        print("- วัดประสิทธิภาพเพื่อเลือกการตั้งค่าที่ดีที่สุด")
        print("- ใช้ metadata ในการกรองและจัดอันดับ")
        
        print("\n🎉 Searching and Querying demo completed successfully!")
        
        return {
            "index": index,
            "basic_results": basic_results,
            "mode_results": mode_results,
            "advanced_results": advanced_results,
            "comparison_results": comparison_results
        }
        
    except Exception as e:
        print(f"❌ Error in demo: {str(e)}")
        return None

if __name__ == "__main__":
    results = main()