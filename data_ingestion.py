import os
import sys
import argparse
from typing import List, Optional
from pathlib import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from tqdm import tqdm

from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

from shared_config import Config


class DataIngestion:
    """數據入庫處理類"""

    def __init__(self):
        """初始化數據入庫系統"""
        # 驗證配置
        Config.validate()

        # 初始化 OpenAI 嵌入模型
        os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
        self.embed_model = OpenAIEmbedding(model=Config.EMBEDDING_MODEL)

        # 初始化語意切割解析器
        self.node_parser = SemanticSplitterNodeParser.from_defaults(
            embed_model=self.embed_model,
            buffer_size=1,
            breakpoint_percentile_threshold=95
        )

        # 初始化 Milvus 向量儲存
        self.vector_store = None
        self._init_milvus()

    def _init_milvus(self):
        """初始化 Milvus 連接"""
        try:
            self.vector_store = MilvusVectorStore(
                host=Config.MILVUS_HOST,
                port=Config.MILVUS_PORT,
                dim=Config.EMBEDDING_DIM,
                collection_name=Config.MILVUS_COLLECTION_NAME,
                overwrite=False  # 不覆蓋現有集合
            )
            print(f"✓ 成功連接到 Milvus: {Config.MILVUS_HOST}:{Config.MILVUS_PORT}")
        except Exception as e:
            print(f"✗ Milvus 連接失敗: {e}")
            sys.exit(1)

    def clear_database(self) -> bool:
        """
        清空 Milvus 數據庫中的所有數據

        Returns:
            是否成功清空
        """
        try:
            print("正在清空 Milvus 數據庫...")

            # 重新初始化向量存儲，使用 overwrite=True 來清空
            self.vector_store = MilvusVectorStore(
                host=Config.MILVUS_HOST,
                port=Config.MILVUS_PORT,
                dim=Config.EMBEDDING_DIM,
                collection_name=Config.MILVUS_COLLECTION_NAME,
                overwrite=True  # 覆蓋現有集合，清空數據
            )

            print("✓ 數據庫清空完成")
            return True

        except Exception as e:
            print(f"✗ 數據庫清空失敗: {e}")
            return False

    def extract_text_from_pdf(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> List[str]:
        """
        從 PDF 提取文字並使用語意分割

        Args:
            pdf_path: PDF 檔案路徑
            page_numbers: 指定頁碼列表

        Returns:
            語意分割後的文字片段列表
        """
        print(f"正在提取 PDF 文字: {pdf_path}")

        try:
            # 提取完整文字
            full_text = ''
            for i, page_layout in enumerate(extract_pages(pdf_path)):
                if page_numbers is not None and i not in page_numbers:
                    continue

                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text = element.get_text()
                        # 處理連字符換行
                        if text.endswith('-\n'):
                            text = text.rstrip('-\n')
                        full_text += text

            if not full_text.strip():
                print("✗ 未提取到任何文字")
                return []

            # 創建 Document 物件用於語意分割
            document = Document(text=full_text.strip())

            # 使用語意分割器分割文字
            nodes = self.node_parser.get_nodes_from_documents([document])

            # 提取分割後的文字片段
            text_chunks = [node.text for node in nodes if node.text.strip()]

            print(f"✓ 語意分割為 {len(text_chunks)} 個文字片段")
            return text_chunks

        except Exception as e:
            print(f"✗ PDF 提取失敗: {e}")
            return []

    def process_documents_from_pdf(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> List[Document]:
        """
        從 PDF 處理文檔

        Args:
            pdf_path: PDF 檔案路徑
            page_numbers: 指定頁碼列表

        Returns:
            Document 物件列表
        """
        paragraphs = self.extract_text_from_pdf(pdf_path, page_numbers)
        documents = []

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                doc = Document(
                    text=paragraph,
                    metadata={
                        "source": pdf_path,
                        "paragraph_id": i,
                        "source_type": "pdf"
                    }
                )
                documents.append(doc)

        print(f"✓ 建立 {len(documents)} 個文檔物件")
        return documents

    def process_documents_from_text(self, texts: List[str], source: str = "text_input") -> List[Document]:
        """
        從文字列表處理文檔

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
                        "text_id": i,
                        "source_type": "text"
                    }
                )
                documents.append(doc)

        print(f"✓ 建立 {len(documents)} 個文檔物件")
        return documents

    def ingest_documents(self, documents: List[Document]) -> bool:
        """
        將文檔入庫到 Milvus

        Args:
            documents: 文檔列表

        Returns:
            是否成功
        """
        try:
            print(f"開始處理 {len(documents)} 個文檔...")

            # 將文檔轉換為節點（文檔已經過語意分割，直接轉換）
            print("正在將文檔轉換為節點...")
            from llama_index.core.schema import TextNode
            nodes = []
            for doc in tqdm(documents, desc="轉換節點", unit="doc"):
                # 直接創建 TextNode，保留文檔的 metadata
                node = TextNode(
                    text=doc.text,
                    metadata=doc.metadata
                )
                nodes.append(node)

            print(f"✓ 轉換為 {len(nodes)} 個節點")

            # 建立儲存上下文
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # 生成嵌入向量並存儲
            print("正在生成嵌入向量並存儲到 Milvus...")
            from llama_index.core import VectorStoreIndex

            # 建立索引，會自動處理嵌入和存儲
            with tqdm(total=len(nodes), desc="存儲節點", unit="node") as pbar:
                # 分批處理以顯示進度
                batch_size = 10
                for i in range(0, len(nodes), batch_size):
                    batch_nodes = nodes[i:i+batch_size]
                    if i == 0:
                        # 第一批創建索引，明確指定 embed_model
                        index = VectorStoreIndex(batch_nodes, storage_context=storage_context, embed_model=self.embed_model)
                    else:
                        # 後續批次添加到現有索引
                        for node in batch_nodes:
                            index.insert_nodes([node])
                    pbar.update(len(batch_nodes))

            print(f"✓ 成功將 {len(nodes)} 個節點存儲到 Milvus")
            return True

        except Exception as e:
            print(f"✗ 文檔入庫失敗: {e}")
            return False

    def ingest_from_directory(self, directory: str, file_pattern: str = "*.pdf") -> bool:
        """
        批次處理目錄中的文件

        Args:
            directory: 目錄路徑
            file_pattern: 文件模式

        Returns:
            是否成功
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            print(f"✗ 目錄不存在: {directory}")
            return False

        # 查找匹配的文件
        files = list(directory_path.glob(file_pattern))
        if not files:
            print(f"✗ 在 {directory} 中未找到匹配 {file_pattern} 的文件")
            return False

        print(f"找到 {len(files)} 個文件待處理")

        success_count = 0
        for file_path in tqdm(files, desc="處理文件", unit="file"):
            print(f"\n處理文件: {file_path.name}")

            if file_path.suffix.lower() == '.pdf':
                documents = self.process_documents_from_pdf(str(file_path))
            else:
                print(f"跳過不支援的文件格式: {file_path.suffix}")
                continue

            if documents and self.ingest_documents(documents):
                success_count += 1
                print(f"✓ {file_path.name} 處理完成")
            else:
                print(f"✗ {file_path.name} 處理失敗")

        print(f"\n批次處理完成: {success_count}/{len(files)} 個文件成功")
        return success_count == len(files)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="RAG 系統數據入庫工具")
    parser.add_argument("--pdf", type=str, help="PDF 檔案路徑")
    parser.add_argument("--directory", type=str, help="批次處理目錄")
    parser.add_argument("--pattern", type=str, default="*.pdf", help="文件匹配模式")
    parser.add_argument("--text", nargs="+", help="直接輸入文字")
    parser.add_argument("--clear", action="store_true", help="清空數據庫")

    args = parser.parse_args()

    # 初始化數據入庫系統
    ingestion = DataIngestion()

    # 檢查是否需要清空數據庫
    if args.clear:
        print("🗑️  清空數據庫選項已啟用")
        if ingestion.clear_database():
            print("✓ 數據庫已清空")
        else:
            print("✗ 數據庫清空失敗")
        return

    if args.pdf:
        # 處理單個 PDF 文件
        documents = ingestion.process_documents_from_pdf(args.pdf)
        if documents:
            success = ingestion.ingest_documents(documents)
            print(f"處理結果: {'成功' if success else '失敗'}")
        else:
            print("未能處理 PDF 文件")

    elif args.directory:
        # 批次處理目錄
        success = ingestion.ingest_from_directory(args.directory, args.pattern)
        print(f"批次處理結果: {'成功' if success else '失敗'}")

    elif args.text:
        # 處理文字輸入
        documents = ingestion.process_documents_from_text(args.text, "command_line")
        if documents:
            success = ingestion.ingest_documents(documents)
            print(f"處理結果: {'成功' if success else '失敗'}")

    else:
        # 示例模式
        print("示例模式：處理測試文字")
        sample_texts = [
            "人工智慧是電腦科學的一個分支，旨在創造能夠執行通常需要人類智慧的任務的系統。",
            "機器學習是人工智慧的子領域，專注於讓電腦系統從資料中學習和改進。",
            "深度學習使用神經網路來模擬人腦的學習過程，在圖像識別和自然語言處理等領域取得了巨大成功。"
        ]

        documents = ingestion.process_documents_from_text(sample_texts, "demo_data")
        if documents:
            success = ingestion.ingest_documents(documents)
            print(f"示例處理結果: {'成功' if success else '失敗'}")


if __name__ == "__main__":
    main()