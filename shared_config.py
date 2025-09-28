import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()


class Config:
    """共享配置類"""

    # OpenAI 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-5"
    LLM_TEMPERATURE = 0.1

    # Milvus 配置
    MILVUS_HOST = "localhost"
    MILVUS_PORT = 19530
    MILVUS_COLLECTION_NAME = "rag_documents"
    EMBEDDING_DIM = 1536  # text-embedding-3-small 維度

    # 文檔處理配置
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    MIN_LINE_LENGTH = 1

    # 查詢配置
    SIMILARITY_TOP_K = 5
    RESPONSE_MODE = "compact"

    # API 配置
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    @classmethod
    def validate(cls):
        """驗證必要的配置"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("請設置 OPENAI_API_KEY 環境變數")

        print("✓ 配置驗證通過")
        return True


# 驗證配置
if __name__ == "__main__":
    Config.validate()