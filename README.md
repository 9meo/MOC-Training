````markdown
# LlamaIndex RAG Workshop (with FAISS, BGE-M3, Thai Documents)

### Workshop สำหรับเรียนรู้ RAG, LlamaIndex, FAISS และ Embedding ภาษาไทย

---

## 1. Clone Repository และเตรียม Environment

```bash
git clone <your-repo-url>
cd <project-folder>
````

---

## 2. สร้าง Python Virtual Environment (แนะนำ Python 3.10)

```bash
python3.10 -m venv ./env
source ./env/bin/activate   # บน Linux/Mac
.\env\Scripts\activate      # บน Windows
```

---

## 3. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

---

## 4. โครงสร้างไฟล์ (File Structure ตัวอย่าง)

```
.
├── requirements.txt
├── 01_embedding_similarity_compare.ipynb
├── 02_chunking_compare.ipynb
├── 03_indexing_multi_files_and_metadata_faiss.ipynb
├── 04_search_compare_chunking_metadata_faiss.ipynb
└── data/
    ├── sample_doc_thai_1.txt
    ├── sample_doc_thai_2.txt
    └── sample_doc_thai_3.txt
```

---

## 5. เริ่มใช้งาน Jupyter Notebook

```bash
jupyter notebook
```

จากนั้นเปิด notebook แต่ละไฟล์ตามลำดับเพื่อทำ workshop

---

## 6. หมายเหตุ

* ถ้าใช้งานบน Windows อาจต้องปรับ path การ activate venv เล็กน้อย
* ควรใช้ Python 3.10 ขึ้นไป
* ตัวอย่างใช้โมเดล BAAI/bge-m3 ใน huggingface (ภาษาไทย)

---

## 7. คำแนะนำเพิ่มเติม

* สามารถแก้ไข/เพิ่มไฟล์ใน data/ เพื่อทดลองกับเอกสารของตนเอง
* ปรับแต่ง chunk size และทดลองตั้งคำถามใหม่ๆ ในแต่ละ notebook

---

```

---

**ถ้าต้องการ README ภาษาไทยล้วน, หรือเพิ่ม section วิธีใช้แต่ละ notebook แจ้งได้เลยครับ!**  
หรือถ้าอยากเพิ่ม badge, github actions, หรืออะไรอีก เพิ่มเติมได้หมดครับ
```
