#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
complete_rag_system_06.py
ระบบ RAG ที่สมบูรณ์สำหรับการใช้งานจริง - Demo Version
"""

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from datetime import datetime
import time
import json
from typing import List, Dict, Any, Optional

class SimpleRAGSystem:
    """
    ระบบ RAG แบบง่ายสำหรับการสาธิต
    """
    
    def __init__(self, embedding_model: str = "BAAI/bge-m3", chunk_size: int = 500):
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.similarity_top_k = 5
        
        print(f"🤖 Initializing Simple RAG System...")
        self._setup_models()
        
        self.index = None
        self.query_engine = None
        self.documents = []
        self.nodes = []
        self.query_history = []
        
        print("✅ RAG System initialized successfully")
    
    def _setup_models(self):
        """ตั้งค่า embedding model"""
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name,
                max_length=8192,
                normalize=True
            )
            Settings.embed_model = self.embed_model
            print(f"📚 Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            print(f"❌ Error setting up models: {str(e)}")
            raise
    
    def load_documents(self, documents: List[Document]) -> int:
        """โหลดเอกสาร"""
        self.documents = documents
        print(f"📄 Loaded {len(self.documents)} documents")
        return len(self.documents)
    
    def process_documents(self) -> int:
        """ประมวลผลเอกสาร"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        # Create node parser
        node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=50
        )
        
        # Parse documents
        self.nodes = node_parser.get_nodes_from_documents(self.documents)
        
        # Add metadata
        for i, node in enumerate(self.nodes):
            node.metadata.update({
                "node_id": f"node_{i}",
                "processed_at": datetime.now().isoformat(),
                "chunk_size": self.chunk_size
            })
        
        print(f"🔧 Processed {len(self.nodes)} nodes")
        return len(self.nodes)
    
    def create_index(self) -> None:
        """สร้าง index"""
        if not self.nodes:
            raise ValueError("No nodes available. Call process_documents() first.")
        
        print("🗂️ Creating vector index...")
        self.index = VectorStoreIndex(nodes=self.nodes)
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            response_mode="compact"
        )
        
        print("✅ Index created successfully")
    
    def query(self, question: str) -> Dict[str, Any]:
        """ถามคำถาม"""
        if not self.query_engine:
            raise ValueError("Index not created. Call create_index() first.")
        
        print(f"\n🔍 Processing query: {question}")
        
        start_time = time.time()
        response = self.query_engine.query(question)
        end_time = time.time()
        
        # Extract source information
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                sources.append({
                    "text": node.text[:200] + "...",
                    "score": getattr(node, 'score', 0),
                    "metadata": node.metadata
                })
        
        result = {
            "question": question,
            "answer": str(response),
            "response_time": end_time - start_time,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.query_history.append(result)
        
        print(f"⏱️ Response time: {end_time - start_time:.3f}s")
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """ดูสถิติระบบ"""
        return {
            "num_documents": len(self.documents),
            "num_nodes": len(self.nodes),
            "num_queries": len(self.query_history),
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "similarity_top_k": self.similarity_top_k,
            "index_created": self.index is not None
        }
    
    def save_history(self, filepath: str) -> None:
        """บันทึกประวัติการใช้งาน"""
        history_data = {
            "system_stats": self.get_stats(),
            "query_history": self.query_history,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 History saved to {filepath}")

class AdvancedFeatures:
    """คุณสมบัติขั้นสูงสำหรับ RAG System"""
    
    def __init__(self, rag_system: SimpleRAGSystem):
        self.rag_system = rag_system
    
    def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """ประมวลผลคำถามหลายข้อพร้อมกัน"""
        print(f"\n📝 Processing {len(queries)} queries in batch...")
        
        results = []
        total_time = 0
        
        for i, query in enumerate(queries, 1):
            print(f"Query {i}/{len(queries)}: {query[:50]}...")
            
            try:
                result = self.rag_system.query(query)
                results.append(result)
                total_time += result['response_time']
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results.append({
                    "question": query,
                    "answer": f"Error: {str(e)}",
                    "response_time": 0,
                    "sources": [],
                    "timestamp": datetime.now().isoformat()
                })
        
        print(f"⏱️ Total time: {total_time:.3f}s")
        print(f"📊 Average time per query: {total_time/len(queries):.3f}s")
        
        return results
    
    def compare_response_modes(self, query: str) -> Dict[str, Any]:
        """เปรียบเทียบ response modes ต่างๆ"""
        print(f"\n🔄 Comparing response modes for: {query}")
        
        modes = ["compact", "tree_summarize", "simple_summarize"]
        comparison_results = {}
        
        for mode in modes:
            try:
                # Create temporary query engine with different mode
                query_engine = self.rag_system.index.as_query_engine(
                    similarity_top_k=self.rag_system.similarity_top_k,
                    response_mode=mode
                )
                
                start_time = time.time()
                response = query_engine.query(query)
                end_time = time.time()
                
                comparison_results[mode] = {
                    "answer": str(response),
                    "response_time": end_time - start_time,
                    "answer_length": len(str(response))
                }
                
                print(f"✅ {mode}: {end_time - start_time:.3f}s, {len(str(response))} chars")
                
            except Exception as e:
                comparison_results[mode] = {"error": str(e)}
                print(f"❌ {mode}: Error - {str(e)}")
        
        return comparison_results
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """วิเคราะห์รูปแบบการใช้งาน"""
        if not self.rag_system.query_history:
            return {"message": "No query history available"}
        
        history = self.rag_system.query_history
        response_times = [q['response_time'] for q in history]
        answer_lengths = [len(q['answer']) for q in history]
        
        # Extract topics
        topics = []
        for query in history:
            text = query['question'].lower()
            if 'machine learning' in text or 'การเรียนรู้ของเครื่อง' in text:
                topics.append('machine_learning')
            elif 'deep learning' in text or 'การเรียนรู้เชิงลึก' in text:
                topics.append('deep_learning')
            elif 'nlp' in text or 'ภาษาธรรมชาติ' in text:
                topics.append('nlp')
            elif 'rag' in text.lower():
                topics.append('rag')
            else:
                topics.append('general')
        
        from collections import Counter
        topic_counts = Counter(topics)
        
        analysis = {
            "total_queries": len(history),
            "avg_response_time": sum(response_times) / len(response_times),
            "max_response_time": max(response_times),
            "min_response_time": min(response_times),
            "avg_answer_length": sum(answer_lengths) / len(answer_lengths),
            "most_common_topics": topic_counts.most_common(3)
        }
        
        return analysis

def create_sample_documents():
    """สร้างเอกสารตัวอย่างสำหรับการทดสอบ"""
    return [
        Document(
            text="""
            การเรียนรู้ของเครื่อง (Machine Learning) เป็นสาขาหนึ่งของปัญญาประดิษฐ์ที่ให้คอมพิวเตอร์สามารถเรียนรู้และปรับปรุงประสิทธิภาพ
            จากประสบการณ์โดยไม่ต้องเขียนโปรแกรมอย่างชัดเจน มีสามประเภทหลัก คือ Supervised Learning ที่ใช้ข้อมูลที่มีป้ายกำกับ
            Unsupervised Learning ที่ค้นหาแพทเทิร์นในข้อมูลที่ไม่มีป้ายกำกับ และ Reinforcement Learning ที่เรียนรู้จากการโต้ตอบกับสภาพแวดล้อม
            อัลกอริทึมที่นิยมใช้ ได้แก่ Linear Regression, Decision Trees, Random Forest, Support Vector Machines และ Neural Networks
            """,
            metadata={"topic": "machine_learning", "difficulty": "beginner", "language": "thai"}
        ),
        Document(
            text="""
            Deep Learning เป็นส่วนหนึ่งของ Machine Learning ที่ใช้โครงข่ายประสาทเทียมหลายชั้น
            สถาปัตยกรรมสำคัญ ได้แก่ CNN สำหรับการประมวลผลภาพ RNN และ LSTM สำหรับข้อมูลลำดับ
            และ Transformer สำหรับการประมวลผลภาษาธรรมชาติ การฝึกฝนต้องการข้อมูลและทรัพยากรจำนวนมาก
            """,
            metadata={"topic": "deep_learning", "difficulty": "intermediate", "language": "thai"}
        ),
        Document(
            text="""
            Natural Language Processing (NLP) เป็นสาขาที่รวมคอมพิวเตอร์ศาสตร์และภาษาศาสตร์
            เพื่อให้คอมพิวเตอร์เข้าใจภาษาของมนุษย์ งานหลัก ได้แก่ Text Classification, NER, Sentiment Analysis
            Machine Translation และ Question Answering เทคนิคสมัยใหม่ใช้ Pre-trained Models เช่น BERT และ GPT
            """,
            metadata={"topic": "nlp", "difficulty": "intermediate", "language": "thai"}
        ),
        Document(
            text="""
            Retrieval-Augmented Generation (RAG) เป็นเทคนิคที่รวม Information Retrieval และ Text Generation
            ประกอบด้วย Retriever ที่ค้นหาเอกสารเกี่ยวข้อง และ Generator ที่สร้างคำตอบจากข้อมูลที่ค้นพบ
            ขั้นตอนคือ สร้าง embeddings, ค้นหาด้วย similarity, แล้วใช้ LLM สร้างคำตอบ
            RAG ช่วยลด hallucination และเพิ่มความแม่นยำของ AI
            """,
            metadata={"topic": "rag", "difficulty": "advanced", "language": "thai"}
        ),
        Document(
            text="""
            Large Language Models (LLMs) เป็น Neural Networks ขนาดใหญ่ที่ผ่านการฝึกฝนด้วยข้อมูลข้อความจำนวนมหาศาล
            โมเดลที่มีชื่อเสียง ได้แก่ GPT, BERT, T5 และ PaLM LLMs สามารถทำงานหลากหลาย
            เช่น การเขียน การแปล การสรุป การตอบคำถาม และการเขียนโค้ด ความสามารถเหล่านี้เกิดจาก emergent properties
            ที่ปรากฏขึ้นเมื่อโมเดลมีขนาดใหญ่พอ
            """,
            metadata={"topic": "llm", "difficulty": "advanced", "language": "thai"}
        )
    ]

def demo_basic_rag():
    """สาธิตระบบ RAG พื้นฐาน"""
    print("🚀 RAG System Basic Demo")
    print("=" * 40)
    
    # 1. Initialize system
    print("\n1️⃣ Initializing RAG System...")
    rag_system = SimpleRAGSystem(chunk_size=400)
    
    # 2. Load documents
    print("\n2️⃣ Loading sample documents...")
    documents = create_sample_documents()
    rag_system.load_documents(documents)
    
    # 3. Process documents
    print("\n3️⃣ Processing documents...")
    rag_system.process_documents()
    
    # 4. Create index
    print("\n4️⃣ Creating index...")
    rag_system.create_index()
    
    # 5. Test queries
    print("\n5️⃣ Testing queries...")
    test_queries = [
        "Machine Learning คืออะไร และมีประเภทใดบ้าง?",
        "ความแตกต่างระหว่าง CNN และ RNN คืออะไร?",
        "RAG ทำงานอย่างไร?",
        "LLM มีความสามารถอะไรบ้าง?"
    ]
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 50)
        
        try:
            result = rag_system.query(query)
            results.append(result)
            
            print(f"💬 Answer: {result['answer'][:300]}...")
            print(f"⏱️ Time: {result['response_time']:.3f}s")
            print(f"📚 Sources: {len(result['sources'])} documents")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return rag_system, results

def demo_advanced_features(rag_system):
    """สาธิตคุณสมบัติขั้นสูง"""
    print("\n" + "=" * 50)
    print("⚡ ADVANCED FEATURES DEMO")
    print("=" * 50)
    
    # Initialize advanced features
    advanced = AdvancedFeatures(rag_system)
    
    # 1. Batch processing
    print("\n1️⃣ Batch Query Processing:")
    batch_queries = [
        "อัลกอริทึม ML มีอะไรบ้าง?",
        "Transformer ทำงานอย่างไร?",
        "NLP ใช้กับภาษาไทยได้ไหม?",
        "RAG vs Fine-tuning ต่างกันอย่างไร?"
    ]
    
    batch_results = advanced.batch_query(batch_queries)
    
    # 2. Response mode comparison
    print("\n2️⃣ Response Mode Comparison:")
    comparison_query = "อธิบาย Deep Learning และการประยุกต์ใช้"
    mode_comparison = advanced.compare_response_modes(comparison_query)
    
    # 3. Query pattern analysis
    print("\n3️⃣ Query Pattern Analysis:")
    analysis = advanced.analyze_query_patterns()
    
    print("📊 Analysis Results:")
    for key, value in analysis.items():
        if key != "message":
            print(f"  {key}: {value}")
    
    return advanced, batch_results, mode_comparison

def demo_system_monitoring(rag_system):
    """สาธิตการตรวจสอบระบบ"""
    print("\n" + "=" * 50)
    print("📊 SYSTEM MONITORING DEMO")
    print("=" * 50)
    
    # 1. System statistics
    print("\n1️⃣ System Statistics:")
    stats = rag_system.get_stats()
    
    for key, value in stats.items():
        print(f"  📈 {key}: {value}")
    
    # 2. Performance analysis
    print("\n2️⃣ Performance Analysis:")
    if rag_system.query_history:
        response_times = [q['response_time'] for q in rag_system.query_history]
        answer_lengths = [len(q['answer']) for q in rag_system.query_history]
        
        print(f"  ⏱️ Average response time: {sum(response_times) / len(response_times):.3f}s")
        print(f"  📏 Average answer length: {sum(answer_lengths) / len(answer_lengths):.0f} chars")
        print(f"  🔥 Fastest query: {min(response_times):.3f}s")
        print(f"  🐌 Slowest query: {max(response_times):.3f}s")
    
    # 3. Save history
    print("\n3️⃣ Saving System History:")
    rag_system.save_history("rag_system_history.json")
    
    return stats

def main():
    """ฟังก์ชันหลักสำหรับ demo ระบบ RAG ที่สมบูรณ์"""
    print("=" * 60)
    print("🤖 COMPLETE RAG SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("🇹🇭 การสาธิตระบบ RAG ที่สมบูรณ์ด้วย LlamaIndex")
    print("=" * 60)
    
    try:
        # Basic RAG demo
        rag_system, basic_results = demo_basic_rag()
        
        # Advanced features demo
        advanced_features, batch_results, mode_comparison = demo_advanced_features(rag_system)
        
        # System monitoring demo
        system_stats = demo_system_monitoring(rag_system)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 RAG SYSTEM DEMO COMPLETED!")
        print("=" * 60)
        
        print("\n✅ What we demonstrated:")
        print("1. 🤖 Complete RAG system setup")
        print("2. 📚 Document processing and indexing")
        print("3. 🔍 Query processing and response generation")
        print("4. ⚡ Advanced features (batch processing, mode comparison)")
        print("5. 📊 System monitoring and performance analysis")
        
        print("\n✅ Key Features:")
        print("- 🌐 Multilingual support (Thai + English)")
        print("- 🚀 Fast embedding with BAAI/bge-m3")
        print("- 🔧 Configurable chunking strategies")
        print("- 📈 Performance monitoring")
        print("- 💾 Query history tracking")
        
        print("\n✅ Files created:")
        print("- 📄 rag_system_history.json: Complete system history")
        
        print(f"\n📊 Final Statistics:")
        print(f"  📚 Documents processed: {system_stats['num_documents']}")
        print(f"  🧩 Nodes created: {system_stats['num_nodes']}")
        print(f"  ❓ Queries processed: {system_stats['num_queries']}")
        
        print("\n🎓 Thank you for trying the RAG Workshop!")
        print("🚀 You're now ready to build your own RAG systems!")
        
        return {
            "rag_system": rag_system,
            "basic_results": basic_results,
            "advanced_features": advanced_features,
            "system_stats": system_stats
        }
        
    except Exception as e:
        print(f"❌ Error in demo: {str(e)}")
        print("Please check your setup and try again.")
        return None

if __name__ == "__main__":
    results = main()