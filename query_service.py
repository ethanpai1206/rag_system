import os
import sys
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.retrievers import VectorIndexRetriever

from shared_config import Config


class QueryRequest(BaseModel):
    """查詢請求模型"""
    question: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    """查詢回應模型"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float


class RelevantDocRequest(BaseModel):
    """相關文檔請求模型"""
    question: str
    top_k: Optional[int] = None


class RelevantDocResponse(BaseModel):
    """相關文檔回應模型"""
    question: str
    documents: List[Dict[str, Any]]
    total_count: int


class QueryService:
    """查詢服務類"""

    def __init__(self):
        """初始化查詢服務"""
        # 驗證配置
        Config.validate()

        # 設置 OpenAI API 金鑰
        os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

        # 初始化嵌入模型
        self.embed_model = OpenAIEmbedding(model=Config.EMBEDDING_MODEL)

        # 初始化 LLM
        self.llm = OpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )

        # 初始化組件
        self.vector_store = None
        self.index = None
        self.query_engine = None
        self.retriever = None

        # 連接 Milvus 並建立索引
        self._init_services()

    def _init_services(self):
        """初始化服務組件"""
        try:
            # 連接到 Milvus
            self.vector_store = MilvusVectorStore(
                host=Config.MILVUS_HOST,
                port=Config.MILVUS_PORT,
                dim=Config.EMBEDDING_DIM,
                collection_name=Config.MILVUS_COLLECTION_NAME,
                overwrite=False
            )
            print(f"✓ 成功連接到 Milvus: {Config.MILVUS_HOST}:{Config.MILVUS_PORT}")

            # 建立儲存上下文
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # 從現有的向量儲存建立索引
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embed_model
            )
            print("✓ 成功載入現有索引")

            # 建立查詢引擎
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=Config.SIMILARITY_TOP_K,
                response_mode=Config.RESPONSE_MODE,
                llm=self.llm
            )

            # 建立檢索器
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=Config.SIMILARITY_TOP_K
            )

            print("✓ 查詢服務初始化完成")

        except Exception as e:
            print(f"✗ 查詢服務初始化失敗: {e}")
            sys.exit(1)

    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        執行問答查詢

        Args:
            question: 問題
            top_k: 檢索文檔數量

        Returns:
            查詢結果
        """
        import time

        start_time = time.time()

        try:
            # 如果指定了 top_k，臨時調整檢索數量
            if top_k and top_k != Config.SIMILARITY_TOP_K:
                query_engine = self.index.as_query_engine(
                    similarity_top_k=top_k,
                    response_mode=Config.RESPONSE_MODE,
                    llm=self.llm
                )
                response = query_engine.query(question)
            else:
                response = self.query_engine.query(question)

            # 處理回應
            answer = str(response)

            # 獲取來源資訊
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_info = {
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": node.score if hasattr(node, 'score') else 0.0,
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }
                    sources.append(source_info)

            processing_time = time.time() - start_time

            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"查詢處理失敗: {str(e)}")

    def get_relevant_documents(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        獲取相關文檔

        Args:
            question: 問題
            top_k: 檢索文檔數量

        Returns:
            相關文檔
        """
        try:
            # 設置檢索數量
            retrieve_count = top_k or Config.SIMILARITY_TOP_K

            # 如果需要臨時調整檢索數量
            if top_k and top_k != Config.SIMILARITY_TOP_K:
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=top_k
                )
                nodes = retriever.retrieve(question)
            else:
                nodes = self.retriever.retrieve(question)

            # 處理檢索結果
            documents = []
            for node in nodes:
                doc_info = {
                    "text": node.text,
                    "score": node.score if hasattr(node, 'score') else 0.0,
                    "metadata": node.metadata if hasattr(node, 'metadata') else {}
                }
                documents.append(doc_info)

            return {
                "question": question,
                "documents": documents,
                "total_count": len(documents)
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文檔檢索失敗: {str(e)}")

    def health_check(self) -> Dict[str, str]:
        """健康檢查"""
        try:
            # 簡單的連接測試
            if self.vector_store and self.index:
                return {"status": "healthy", "message": "服務正常運行"}
            else:
                return {"status": "unhealthy", "message": "服務組件未正確初始化"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"健康檢查失敗: {str(e)}"}


# 建立 FastAPI 應用
app = FastAPI(
    title="RAG 查詢服務 API",
    description="基於 Milvus 和 OpenAI 的檢索增強生成服務",
    version="1.0.0"
)

# 添加 CORS 中介軟體
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該限制來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化查詢服務
query_service = QueryService()


@app.get("/")
async def root():
    """根路徑"""
    return {"message": "RAG 查詢服務正在運行", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return query_service.health_check()


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """問答查詢端點"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="問題不能為空")

    result = query_service.query(request.question, request.top_k)
    return QueryResponse(**result)


@app.post("/relevant-docs", response_model=RelevantDocResponse)
async def relevant_docs_endpoint(request: RelevantDocRequest):
    """相關文檔檢索端點"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="問題不能為空")

    result = query_service.get_relevant_documents(request.question, request.top_k)
    return RelevantDocResponse(**result)


@app.get("/stats")
async def get_stats():
    """獲取服務統計資訊"""
    try:
        # 這裡可以添加更多統計資訊
        return {
            "milvus_host": Config.MILVUS_HOST,
            "milvus_port": Config.MILVUS_PORT,
            "collection_name": Config.MILVUS_COLLECTION_NAME,
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.LLM_MODEL,
            "default_top_k": Config.SIMILARITY_TOP_K
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取統計資訊失敗: {str(e)}")


def main():
    """主函數"""
    print("啟動 RAG 查詢服務...")
    print(f"API 文檔: http://{Config.API_HOST}:{Config.API_PORT}/docs")

    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()