#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_metadata_extraction.py
Metadata Extraction สำหรับปรับปรุงการจัดทำดัชนีและความเข้าใจเอกสาร
"""

from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import json
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class ThaiKeywordExtractor(BaseExtractor):
    """
    Custom extractor สำหรับสกัด keywords ภาษาไทย
    """
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm, **kwargs)
        
    async def aextract(self, nodes):
        metadata_list = []
        
        for node in nodes:
            # Simple keyword extraction for Thai text
            keywords = self._extract_thai_keywords(node.text)
            metadata_list.append({"thai_keywords": keywords})
            
        return metadata_list
    
    def extract(self, nodes):
        """Synchronous version of extract"""
        metadata_list = []
        
        for node in nodes:
            keywords = self._extract_thai_keywords(node.text)
            metadata_list.append({"thai_keywords": keywords})
            
        return metadata_list
    
    def _extract_thai_keywords(self, text: str) -> List[str]:
        """
        Simple Thai keyword extraction
        """
        # Common Thai technical terms
        tech_terms = [
            "ปัญญาประดิษฐ์", "Machine Learning", "Deep Learning", "Neural Networks",
            "การเรียนรู้ของเครื่อง", "การเรียนรู้เชิงลึก", "โครงข่ายประสาทเทียม",
            "ประมวลผลภาษาธรรมชาติ", "NLP", "อัลกอริทึม", "โมเดล", "ข้อมูล",
            "CNN", "RNN", "LSTM", "Transformer", "BERT", "GPT", "RAG"
        ]
        
        found_keywords = []
        for term in tech_terms:
            if term in text:
                found_keywords.append(term)
                
        return found_keywords

def setup_models():
    """ตั้งค่าโมเดลที่จำเป็น"""
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        max_length=8192,
        normalize=True
    )

    # Setup LLM (ในการใช้งานจริงให้ใส่ API key)
    try:
        # llm = OpenAI(temperature=0.1)
        llm = Ollama(model="gemma3n:e2b", request_timeout=60.0)
    except:
        print("Warning: OpenAI API not available, using mock LLM")
        llm = None

    Settings.embed_model = embed_model
    Settings.llm = llm
    
    return embed_model, llm

def create_sample_documents():
    """สร้างเอกสารตัวอย่างสำหรับ metadata extraction"""
    documents = [
        Document(
            text="""
            การเรียนรู้ของเครื่อง (Machine Learning) เป็นสาขาหนึ่งของปัญญาประดิษฐ์ที่มุ่งเน้นการพัฒนาอัลกอริทึมที่สามารถเรียนรู้จากข้อมูล 
            โดยไม่ต้องเขียนโปรแกรมอย่างชัดเจน มีหลายประเภท เช่น Supervised Learning ที่ใช้ข้อมูลที่มีป้ายกำกับ 
            Unsupervised Learning ที่ค้นหาแพทเทิร์นในข้อมูลที่ไม่มีป้ายกำกับ และ Reinforcement Learning ที่เรียนรู้จากการทดลองและความคิดเห็น
            """,
            metadata={"source": "ML_basics.pdf", "page": 1}
        ),
        Document(
            text="""
            Neural Networks เป็นโมเดลที่ได้รับแรงบันดาลใจจากระบบประสาทของสมองมนุษย์ ประกอบด้วยชั้นของ neurons ที่เชื่อมต่อกัน 
            การเรียนรู้เชิงลึก (Deep Learning) ใช้ neural networks ที่มีหลายชั้น เรียกว่า Deep Neural Networks 
            ซึ่งมีประสิทธิภาพสูงในการแก้ปัญหาซับซ้อน เช่น การรู้จำภาพ การประมวลผลภาษาธรรมชาติ และการสร้าง content
            """,
            metadata={"source": "neural_networks.pdf", "page": 1}
        ),
        Document(
            text="""
            Natural Language Processing (NLP) เป็นสาขาที่รวมศาสตร์คอมพิวเตอร์ ปัญญาประดิษฐ์ และภาษาศาสตร์ 
            เพื่อช่วยให้คอมพิวเตอร์เข้าใจและประมวลผลภาษาของมนุษย์ งานสำคัญใน NLP ได้แก่ การแปลภาษา การสรุปข้อความ 
            การวิเคราะห์ความรู้สึก การตอบคำถาม และการสร้างข้อความ เทคโนโลยี NLP ถูกนำไปใช้ใน chatbots, search engines และ voice assistants
            """,
            metadata={"source": "nlp_overview.pdf", "page": 1}
        )
    ]

    print(f"Created {len(documents)} documents for metadata extraction")
    return documents

def setup_metadata_extractors(llm):
    """
    ตั้งค่า metadata extractors ต่างๆ
    """
    extractors = []
    
    # เพิ่ม extractors ตามที่มี LLM หรือไม่
    if llm:
        extractors.extend([
            # Title extractor - สกัดชื่อเรื่องของเอกสาร
            TitleExtractor(nodes=5, llm=llm),
            
            # Summary extractor - สร้างสรุปเนื้อหา
            SummaryExtractor(summaries=["prev", "self"], llm=llm),
            
            # Questions answered extractor - สกัดคำถามที่เอกสารตอบได้
            QuestionsAnsweredExtractor(questions=3, llm=llm),
            
            # Keyword extractor - สกัดคำสำคัญ
            KeywordExtractor(keywords=10, llm=llm),
        ])
    
    # Custom Thai keyword extractor (ไม่ต้องใช้ LLM)
    extractors.append(ThaiKeywordExtractor(llm=llm))
    
    return extractors

def process_documents_sync(documents: List[Document], embed_model, llm):
    """
    ประมวลผลเอกสารแบบ sync
    """
    # Simple node parsing
    node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    # Apply Thai keyword extraction
    thai_extractor = ThaiKeywordExtractor()
    thai_metadata = thai_extractor.extract(nodes)
    
    # Add metadata to nodes
    for i, node in enumerate(nodes):
        # Extract simple keywords
        keywords = thai_metadata[i]["thai_keywords"]
        
        # Add enhanced metadata
        node.metadata.update({
            "node_id": f"node_{i}",
            "thai_keywords": keywords,
            "content_type": "technical",
            "language": "thai",
            "char_count": len(node.text),
            "word_count": len(node.text.split()),
            "estimated_reading_time": len(node.text) // 200,  # rough estimate
            "processing_timestamp": pd.Timestamp.now().isoformat()
        })
    
    return nodes

def extract_advanced_metadata(node):
    """
    สกัด metadata ขั้นสูงสำหรับเอกสาร
    """
    text = node.text
    
    # Content analysis
    sentences = text.split('.')
    words = text.split()
    
    # Technical term density
    tech_terms = ["เทคโนโลยี", "อัลกอริทึม", "โมเดล", "ข้อมูล", "ระบบ", "การเรียนรู้"]
    tech_density = sum(1 for word in words if any(term in word for term in tech_terms)) / len(words) if words else 0
    
    # Complexity score (based on sentence length)
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    complexity_score = min(avg_sentence_length / 20, 1.0) if avg_sentence_length > 0 else 0  # normalized to 0-1
    
    # Topic classification (simple rule-based)
    if "Machine Learning" in text or "การเรียนรู้ของเครื่อง" in text:
        topic = "machine_learning"
    elif "Neural Network" in text or "โครงข่ายประสาทเทียม" in text:
        topic = "neural_networks"
    elif "NLP" in text or "ภาษาธรรมชาติ" in text:
        topic = "nlp"
    else:
        topic = "general_ai"
    
    return {
        "tech_density": tech_density,
        "complexity_score": complexity_score,
        "topic": topic,
        "sentence_count": len([s for s in sentences if s.strip()]),
        "word_count": len(words),
        "avg_sentence_length": avg_sentence_length
    }

def analyze_metadata(nodes):
    """
    วิเคราะห์ metadata ที่สกัดออกมา
    """
    print("Metadata Analysis:")
    print("=" * 50)
    
    for i, node in enumerate(nodes[:3]):  # แสดงแค่ 3 nodes แรก
        print(f"\nNode {i+1}:")
        print(f"Text preview: {node.text[:100]}...")
        print(f"Metadata: {json.dumps(node.metadata, ensure_ascii=False, indent=2)}")
        print("-" * 30)
    
    # Create summary DataFrame
    metadata_summary = []
    for node in nodes:
        metadata_summary.append({
            'node_id': node.metadata.get('node_id', 'unknown'),
            'char_count': node.metadata.get('char_count', 0),
            'keywords_count': len(node.metadata.get('thai_keywords', [])),
            'source': node.metadata.get('source', 'unknown'),
            'content_type': node.metadata.get('content_type', 'unknown'),
            'reading_time': node.metadata.get('estimated_reading_time', 0),
            'topic': node.metadata.get('topic', 'unknown'),
            'tech_density': node.metadata.get('tech_density', 0),
            'complexity_score': node.metadata.get('complexity_score', 0)
        })
    
    df = pd.DataFrame(metadata_summary)
    print(f"\nMetadata Summary:")
    print(df.to_string(index=False))
    
    return df

def visualize_metadata_insights(nodes):
    """
    แสดงภาพข้อมูลเชิงลึกจาก metadata
    """
    # Prepare data
    data = []
    for node in nodes:
        data.append({
            'tech_density': node.metadata.get('tech_density', 0),
            'complexity_score': node.metadata.get('complexity_score', 0),
            'topic': node.metadata.get('topic', 'unknown'),
            'word_count': node.metadata.get('word_count', 0),
            'keywords_count': len(node.metadata.get('thai_keywords', [])),
            'avg_sentence_length': node.metadata.get('avg_sentence_length', 0)
        })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data available for visualization")
        return df
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Document Metadata Analysis', fontsize=16)
    
    # Technical density by topic
    if 'topic' in df.columns and not df['topic'].isna().all():
        try:
            sns.boxplot(data=df, x='topic', y='tech_density', ax=axes[0,0])
            axes[0,0].set_title('Technical Density by Topic')
            axes[0,0].set_xlabel('Topic')
            axes[0,0].set_ylabel('Technical Density')
        except:
            axes[0,0].set_title('Technical Density by Topic (No Data)')
    
    # Complexity vs Word Count
    sns.scatterplot(data=df, x='word_count', y='complexity_score', hue='topic', ax=axes[0,1])
    axes[0,1].set_title('Complexity vs Word Count')
    axes[0,1].set_xlabel('Word Count')
    axes[0,1].set_ylabel('Complexity Score')
    
    # Keywords distribution
    if 'topic' in df.columns:
        try:
            sns.barplot(data=df, x='topic', y='keywords_count', ax=axes[1,0])
            axes[1,0].set_title('Keywords Count by Topic')
            axes[1,0].set_xlabel('Topic')
            axes[1,0].set_ylabel('Keywords Count')
        except:
            axes[1,0].set_title('Keywords Count by Topic (No Data)')
    
    # Average sentence length by topic
    if 'topic' in df.columns:
        try:
            sns.barplot(data=df, x='topic', y='avg_sentence_length', ax=axes[1,1])
            axes[1,1].set_title('Average Sentence Length by Topic')
            axes[1,1].set_xlabel('Topic')
            axes[1,1].set_ylabel('Average Sentence Length')
        except:
            axes[1,1].set_title('Average Sentence Length by Topic (No Data)')
    
    plt.tight_layout()
    plt.show()
    
    return df

def save_nodes_with_metadata(nodes, filename="nodes_with_metadata.json"):
    """
    บันทึก nodes พร้อม metadata เพื่อใช้ในขั้นตอนต่อไป
    """
    nodes_data = []
    for node in nodes:
        nodes_data.append({
            "text": node.text,
            "metadata": node.metadata,
            "id": node.id_
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nodes_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(nodes)} nodes to {filename}")

def main():
    """ฟังก์ชันหลักสำหรับทดสอบ metadata extraction"""
    print("=" * 60)
    print("METADATA EXTRACTION DEMO")
    print("=" * 60)
    
    # Setup models
    embed_model, llm = setup_models()
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Process documents with metadata extraction
    print("\nProcessing documents with metadata extraction...")
    nodes_with_metadata = process_documents_sync(documents, embed_model, llm)
    print(f"Created {len(nodes_with_metadata)} nodes with metadata")
    
    # Apply advanced metadata extraction
    print("\nApplying advanced metadata extraction...")
    for node in nodes_with_metadata:
        advanced_metadata = extract_advanced_metadata(node)
        node.metadata.update(advanced_metadata)
    
    print("Advanced metadata extraction completed!")
    
    # Analyze extracted metadata
    metadata_df = analyze_metadata(nodes_with_metadata)
    
    # Visualize metadata insights
    print("\nGenerating metadata visualization...")
    metadata_insights_df = visualize_metadata_insights(nodes_with_metadata)
    
    # Save processed nodes
    save_nodes_with_metadata(nodes_with_metadata)
    
    # Summary
    print("\n" + "=" * 60)
    print("สรุป Metadata Extraction")
    print("=" * 60)
    print("\nประโยชน์ของ Metadata Extraction:")
    print("1. ปรับปรุงการค้นหา: ใช้ metadata ในการกรองและจัดอันดับผลลัพธ์")
    print("2. เข้าใจเนื้อหา: สกัดข้อมูลสำคัญอย่างอัตโนมัติ")
    print("3. จัดหมวดหมู่: แยกประเภทเอกสารตามเนื้อหา")
    print("4. คุณภาพเนื้อหา: วัดความซับซ้อนและความเหมาะสมของเนื้อหา")
    print("\nExtractors ที่ใช้:")
    print("- SummaryExtractor: สร้างสรุปเนื้อหา")
    print("- QuestionsAnsweredExtractor: สกัดคำถามที่ตอบได้")
    print("- TitleExtractor: สกัดชื่อเรื่อง")
    print("- KeywordExtractor: สกัดคำสำคัญ")
    print("- Custom Thai Extractors: สกัดข้อมูลเฉพาะสำหรับภาษาไทย")
    
    return nodes_with_metadata, metadata_df

if __name__ == "__main__":
    nodes, metadata_df = main()