import os
from typing import List, Optional
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.milvus import MilvusVectorStore

# 載入 .env 檔案
load_dotenv()

class RAGSystem:
    """基於 LlamaIndex 的 RAG 系統"""

    def __init__(self, openai_api_key: Optional[str] = None, use_milvus: bool = False):
        """
        初始化 RAG 系統

        Args:
            openai_api_key: OpenAI API 金鑰，若未提供則從環境變數讀取
            use_milvus: 是否使用 Milvus 作為向量資料庫
        """
        # 從 .env 檔案或環境變數獲取 API 金鑰
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("請在 .env 檔案中設置 OPENAI_API_KEY 或直接傳入 API 金鑰")

        # 設置 OpenAI API 金鑰
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # 配置嵌入模型 (text-embedding-3-small)
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small"
        )

        # 配置 LLM 模型 (GPT-4)
        self.llm = OpenAI(
            model="gpt-5",
            temperature=0.1
        )

        # 設置全局配置
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        # 初始化其他組件
        self.node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
        self.index = None
        self.query_engine = None
        self.use_milvus = use_milvus
        self.vector_store = None

        # 如果使用 Milvus，初始化向量儲存
        if self.use_milvus:
            try:
                self.vector_store = MilvusVectorStore(
                    host="localhost",
                    port=19530,
                    dim=1536,  # text-embedding-3-small 維度
                    collection_name="rag_documents"
                )
                print("Milvus 向量儲存初始化成功")
            except Exception as e:
                print(f"Milvus 向量儲存初始化失敗: {e}")
                self.use_milvus = False

    def extract_text_from_pdf(self, filename: str, page_numbers: Optional[List[int]] = None, min_line_length: int = 1) -> List[str]:
        """
        從 PDF 文件中提取文字

        Args:
            filename: PDF 檔案路徑
            page_numbers: 指定頁碼列表，None 表示全部頁面
            min_line_length: 最小行長度

        Returns:
            提取的段落列表
        """
        paragraphs = []
        buffer = ''
        full_text = ''

        # 提取全部文本
        for i, page_layout in enumerate(extract_pages(filename)):
            # 如果指定了頁碼範圍，跳過範圍外的頁
            if page_numbers is not None and i not in page_numbers:
                continue
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    full_text += element.get_text() + '\n'

        # 按空行分隔，將文本重新組織成段落
        lines = full_text.split('\n')
        for text in lines:
            if len(text) >= min_line_length:
                buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
            elif buffer:
                paragraphs.append(buffer)
                buffer = ''
        if buffer:
            paragraphs.append(buffer)

        return paragraphs

    def load_documents_from_pdf(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> List[Document]:
        """
        從 PDF 載入文檔

        Args:
            pdf_path: PDF 檔案路徑
            page_numbers: 指定頁碼列表

        Returns:
            Document 物件列表
        """
        paragraphs = self.extract_text_from_pdf(pdf_path, page_numbers)
        documents = []

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():  # 跳過空段落
                doc = Document(
                    text=paragraph,
                    metadata={
                        "source": pdf_path,
                        "paragraph_id": i
                    }
                )
                documents.append(doc)

        return documents

    def load_documents_from_text(self, texts: List[str], source: str = "text_input") -> List[Document]:
        """
        從文字列表載入文檔

        Args:
            texts: 文字列表
            source: 資料來源標識

        Returns:
            Document 物件列表
        """
        documents = []
        for i, text in enumerate(texts):
            if text.strip():
                doc = Document(
                    text=text,
                    metadata={
                        "source": source,
                        "text_id": i
                    }
                )
                documents.append(doc)

        return documents

    def build_index(self, documents: List[Document]):
        """
        建立向量索引

        Args:
            documents: 文檔列表
        """
        print(f"正在建立索引，共 {len(documents)} 個文檔...")

        # 解析文檔為節點
        nodes = self.node_parser.get_nodes_from_documents(documents)
        print(f"解析為 {len(nodes)} 個節點")

        if self.use_milvus and self.vector_store:
            # 使用 Milvus 向量儲存
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex(nodes, storage_context=storage_context)
            print("已使用 Milvus 建立索引")
        else:
            # 使用 LlamaIndex 原生方式
            self.index = VectorStoreIndex(nodes)
            print("已使用記憶體建立索引")

        # 建立查詢引擎
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )

        print("索引建立完成！")

    def query(self, question: str) -> str:
        """
        查詢問題

        Args:
            question: 問題

        Returns:
            回答
        """
        if self.query_engine is None:
            raise ValueError("請先建立索引後再進行查詢")

        print(f"正在查詢: {question}")
        response = self.query_engine.query(question)
        return str(response)

    def get_relevant_documents(self, question: str, top_k: int = 5) -> List[str]:
        """
        獲取相關文檔

        Args:
            question: 問題
            top_k: 返回前 k 個相關文檔

        Returns:
            相關文檔列表
        """
        if self.index is None:
            raise ValueError("請先建立索引")

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )

        nodes = retriever.retrieve(question)
        return [node.text for node in nodes]


def main():
    """主函數示例"""
    # 初始化 RAG 系統（會自動從 .env 檔案讀取 API 金鑰）
    # 設置 use_milvus=True 來使用 Milvus 向量資料庫
    try:
        rag = RAGSystem(use_milvus=False)  # 改為 True 可使用 Milvus
    except ValueError as e:
        print(f"錯誤: {e}")
        return

    # 示例 1: 從 PDF 載入文檔
    # pdf_path = "example.pdf"
    # if os.path.exists(pdf_path):
    #     documents = rag.load_documents_from_pdf(pdf_path)
    #     rag.build_index(documents)
    #
    #     # 查詢示例
    #     answer = rag.query("這份文檔的主要內容是什麼？")
    #     print(f"回答: {answer}")

    # 示例 2: 從文字載入文檔
    sample_texts = [
        "人工智慧是電腦科學的一個分支，旨在創造能夠執行通常需要人類智慧的任務的系統。",
        "機器學習是人工智慧的子領域，專注於讓電腦系統從資料中學習和改進。",
        "深度學習使用神經網路來模擬人腦的學習過程，在圖像識別和自然語言處理等領域取得了巨大成功。"
    ]

    documents = rag.load_documents_from_text(sample_texts, "AI_knowledge")
    rag.build_index(documents)

    # 查詢示例
    questions = [
        "什麼是人工智慧？",
        "機器學習和深度學習有什麼關係？",
        "深度學習有哪些應用領域？"
    ]

    for question in questions:
        answer = rag.query(question)
        print(f"\n問題: {question}")
        print(f"回答: {answer}")
        print("-" * 50)


if __name__ == "__main__":
    main()