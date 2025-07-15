from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
import uvicorn

from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.llms.ollama import Ollama
# ---------- CONFIG ----------
EMBED_DIM = 1024                   # BAAI/bge‑m3
FAISS_INDEX_PATH = "./storage/"
API_KEY = "supersecretapikey"      # ตั้งเป็น env ในโปรดักชัน
# -----------------------------
llm = Ollama(model="gemma3n:e2b", request_timeout=60.0)
app = FastAPI()

# --- Load index once at startup ---
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

faiss_store = FaissVectorStore.from_persist_dir(FAISS_INDEX_PATH)

storage_ctx = StorageContext.from_defaults(vector_store=faiss_store)
index = load_index_from_storage(storage_context=storage_ctx)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
engine = RetrieverQueryEngine(
    retriever=retriever,
    input_prompt=DEFAULT_SIMPLE_INPUT_PROMPT
)

# ---------- Schemas ---------- #
class RetrievalSetting(BaseModel):
    top_k: int = 3
    score_threshold: float = 0.0   # ค่าเริ่มต้น = เอาทุกผล

class DifyRetrievalRequest(BaseModel):
    knowledge_id: str
    query: str
    retrieval_setting: RetrievalSetting
    metadata_condition: dict | None = None  # ยังไม่ใช้งาน แต่รับไว้ตามสเปก

# ---------- Helpers ---------- #
def verify_api_key(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": 1001,
                    "error_msg": "Invalid Authorization header format."}
        )
    token = auth.split("Bearer ")[1].strip()
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": 1002,
                    "error_msg": "Authorization failed"}
        )

# ---------- Endpoint ---------- #
@app.post("/retrieval")
async def retrieval(request: Request, body: DifyRetrievalRequest):
    verify_api_key(request)

    # ตั้งค่า top_k และ threshold จากรีเควสต์
    retriever.similarity_top_k = body.retrieval_setting.top_k
    response = engine.query(body.query)

    records = []
    for node in response.source_nodes:
        # ตัดผลลัพธ์ตาม score_threshold
        if node.score is not None and node.score < body.retrieval_setting.score_threshold:
            continue

        records.append({
            "content": node.get_content(),
            "score": round(float(node.score), 4) if node.score is not None else 1.0,
            "title": node.metadata.get("title", "unknown"),
            "metadata": node.metadata or {}
        })

    return {"records": records}

@app.get("/")
def status():
    return {"msg": "Dify‑compatible Retrieval API running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)


# curl -X POST http://localhost:8080/retrieval \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer supersecretapikey" \
#   -d '{
#         "knowledge_id": "demo-knowledge",
#         "query": "AI สำคัญกับประเทศไทยอย่างไร",
#         "retrieval_setting": {
#             "top_k": 3,
#             "score_threshold": 0.5
#         }
#       }'