import os
import sys
import argparse
from typing import List, Dict, Any, Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from mxbai_rerank import MxbaiRerankV2

from shared_config import Config
from logging_config import get_query_logger


class LocalQuerySystem:
    """本地查詢系統類"""

    def __init__(self):
        """初始化本地查詢系統"""
        # 初始化 logger（只記錄到文件，不顯示在 console）
        self.query_logger = get_query_logger(enable_console=False)

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

        # 自定義 prompt 模板
        self.custom_template = PromptTemplate(
            """你是一个问答机器人。
                你的任务是根据下述给定的已知信息回答用户问题。

                已知信息:
                {context_str}

                用户问：
                {query_str}

                如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
                请不要输出已知信息中不包含的信息或答案。
                请用中文回答用户问题。
                """
        )

        # 初始化重排模型
        self.reranker = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2")

        # 初始化組件
        self.vector_store = None
        self.index = None
        self.query_engine = None
        self.retriever = None

        # 連接 Milvus 並建立索引
        self._init_system()

    def _init_system(self):
        """初始化系統組件"""
        try:
            print("正在連接 Milvus...")
            # 連接到 Milvus
            self.vector_store = MilvusVectorStore(
                host=Config.MILVUS_HOST,
                port=Config.MILVUS_PORT,
                dim=Config.EMBEDDING_DIM,
                collection_name=Config.MILVUS_COLLECTION_NAME,
                overwrite=False
            )
            print(f"✓ 成功連接到 Milvus: {Config.MILVUS_HOST}:{Config.MILVUS_PORT}")

            print("正在載入索引...")
            # 從現有的向量儲存建立索引
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embed_model
            )
            print("✓ 成功載入索引")

            # 建立查詢引擎（使用自定義 prompt 模板）
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=Config.SIMILARITY_TOP_K,
                response_mode=Config.RESPONSE_MODE,
                llm=self.llm,
                text_qa_template=self.custom_template
            )

            # 建立檢索器
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=Config.SIMILARITY_TOP_K
            )

            print("✓ 本地查詢系統初始化完成\n")

        except Exception as e:
            print(f"✗ 系統初始化失敗: {e}")
            print("\n請確保：")
            print("1. Milvus 服務已啟動")
            print("2. 已有數據在 Milvus 中（使用 document_indexing.py 先入庫數據）")
            print("3. OpenAI API 金鑰設置正確")
            sys.exit(1)

    def query(self, question: str, top_k: Optional[int] = None, show_sources: bool = True, use_rerank: bool = False) -> Dict[str, Any]:
        """
        執行問答查詢

        Args:
            question: 問題
            top_k: 檢索文檔數量
            show_sources: 是否顯示來源
            use_rerank: 是否使用重排模型

        Returns:
            查詢結果
        """
        import time

        start_time = time.time()

        try:
            print(f"🔍 查詢問題: {question}")

            if use_rerank:
                # 使用重排模型的查詢流程
                print("🔄 使用重排模型進行查詢...")

                # 先檢索較多的文檔用於重排
                retrieve_count = (top_k or Config.SIMILARITY_TOP_K) * 2
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=retrieve_count
                )
                nodes = retriever.retrieve(question)

                # 準備文檔列表用於重排
                documents = [node.text for node in nodes]

                # 使用重排模型
                rerank_results = self.reranker.rank(
                    question,
                    documents,
                    return_documents=True,
                    top_k=top_k or Config.SIMILARITY_TOP_K
                )

                print(f"✓ 重排完成，選出前 {len(rerank_results)} 個最相關文檔")

                # 重新構建上下文
                reranked_context = "\n\n".join([result.document for result in rerank_results])

                # 直接使用 LLM 生成回答
                prompt = self.custom_template.format(
                    context_str=reranked_context,
                    query_str=question
                )

                response_text = self.llm.complete(prompt).text

                # 模擬回應物件
                class MockResponse:
                    def __init__(self, text, source_nodes):
                        self.text = text
                        self.source_nodes = source_nodes

                    def __str__(self):
                        return self.text

                # 根據重排結果重新組織 source_nodes
                reranked_nodes = []
                for i, result in enumerate(rerank_results):
                    # 找到對應的原始節點
                    for node in nodes:
                        if node.text == result.document:
                            node.score = result.score
                            reranked_nodes.append(node)
                            break

                response = MockResponse(response_text, reranked_nodes)

            else:
                # 原有的查詢流程（不使用重排）
                if top_k and top_k != Config.SIMILARITY_TOP_K:
                    query_engine = self.index.as_query_engine(
                        similarity_top_k=top_k,
                        response_mode=Config.RESPONSE_MODE,
                        llm=self.llm,
                        text_qa_template=self.custom_template
                    )
                    response = query_engine.query(question)
                else:
                    response = self.query_engine.query(question)

            # 處理回應
            answer = str(response)
            processing_time = time.time() - start_time

            print(f"\n💬 回答:")
            print(answer)
            print(f"\n⏱️  處理時間: {processing_time:.2f} 秒")

            # 處理來源資訊（只記錄到 log 文件，不在 console 顯示）
            sources = []
            if show_sources and hasattr(response, 'source_nodes'):
                # 組織完整的來源資訊用於 log 記錄
                sources_log = "\n📚 參考來源:\n"

                for i, node in enumerate(response.source_nodes, 1):
                    score = node.score if hasattr(node, 'score') else 0.0
                    metadata = node.metadata if hasattr(node, 'metadata') else {}

                    sources_log += f"\n{i}. 相似度分數: {score:.4f}\n"
                    if 'source' in metadata:
                        sources_log += f"   來源: {metadata['source']}\n"
                    if 'paragraph_id' in metadata:
                        sources_log += f"   段落 ID: {metadata['paragraph_id']}\n"

                    # 記錄完整內容到 log
                    sources_log += f"   內容: {node.text}\n"

                    source_info = {
                        "text": node.text,
                        "score": score,
                        "metadata": metadata
                    }
                    sources.append(source_info)

                # 只記錄到 log 文件，不在 console 顯示
                self.query_logger.log_info(sources_log)

            result = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time
            }

            # 記錄查詢結果到 log 文件
            self.query_logger.log_query_result(result)

            return result

        except Exception as e:
            print(f"✗ 查詢失敗: {e}")
            return {
                "question": question,
                "answer": f"查詢失敗: {e}",
                "sources": [],
                "processing_time": 0.0
            }

    def get_relevant_documents(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        獲取相關文檔

        Args:
            question: 問題
            top_k: 檢索文檔數量

        Returns:
            相關文檔列表
        """
        try:
            print(f"🔍 檢索相關文檔: {question}")

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
            print(f"\n📄 找到 {len(nodes)} 個相關文檔:")

            for i, node in enumerate(nodes, 1):
                score = node.score if hasattr(node, 'score') else 0.0
                metadata = node.metadata if hasattr(node, 'metadata') else {}

                print(f"\n{i}. 相似度分數: {score:.4f}")
                if 'source' in metadata:
                    print(f"   來源: {metadata['source']}")
                if 'paragraph_id' in metadata:
                    print(f"   段落 ID: {metadata['paragraph_id']}")

                # 顯示文本片段
                print(f"   內容: {node.text}")
                print("-" * 80)

                doc_info = {
                    "text": node.text,
                    "score": score,
                    "metadata": metadata
                }
                documents.append(doc_info)

            return documents

        except Exception as e:
            print(f"✗ 文檔檢索失敗: {e}")
            return []

    def interactive_mode(self):
        """交互式查詢模式"""
        print("🎯 進入交互式查詢模式")
        print("輸入 'quit' 或 'exit' 退出")
        print("輸入 'docs <問題>' 僅檢索相關文檔")
        print("輸入 'help' 查看命令說明")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n請輸入您的問題: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 再見！")
                    break

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                if user_input.lower().startswith('docs '):
                    # 僅檢索文檔模式
                    question = user_input[5:].strip()
                    if question:
                        self.get_relevant_documents(question)
                    else:
                        print("❌ 請提供要檢索的問題")
                    continue

                # 正常問答模式
                self.query(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 再見！")
                break
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")

    def _show_help(self):
        """顯示幫助信息"""
        print("\n📖 命令說明:")
        print("• 直接輸入問題 - 進行問答查詢")
        print("• docs <問題> - 僅檢索相關文檔，不生成回答")
        print("• help - 顯示此幫助信息")
        print("• quit/exit/q - 退出程序")

    def batch_query(self, questions: List[str], output_file: Optional[str] = None):
        """
        批次查詢

        Args:
            questions: 問題列表
            output_file: 輸出文件路徑
        """
        results = []

        print(f"🔄 開始批次查詢 {len(questions)} 個問題...")

        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] 處理中...")
            result = self.query(question, show_sources=False)
            results.append(result)
            print("=" * 80)

        if output_file:
            try:
                import json
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n✅ 結果已保存到: {output_file}")
            except Exception as e:
                print(f"❌ 保存文件失敗: {e}")

        return results


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="本地 RAG 查詢工具")
    parser.add_argument("--question", "-q", type=str, help="單次查詢問題")
    parser.add_argument("--docs", "-d", type=str, help="檢索相關文檔")
    parser.add_argument("--top-k", "-k", type=int, help="檢索文檔數量")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式模式")
    parser.add_argument("--batch", "-b", nargs="+", help="批次查詢問題列表")
    parser.add_argument("--output", "-o", type=str, help="輸出文件路徑（批次模式）")
    parser.add_argument("--no-sources", action="store_true", help="不顯示來源信息")
    parser.add_argument("--no-rerank", action="store_true", help="不使用重排模型")

    args = parser.parse_args()

    # 初始化查詢系統
    try:
        query_system = LocalQuerySystem()
    except Exception as e:
        print(f"系統初始化失敗: {e}")
        return

    if args.question:
        # 單次查詢模式
        query_system.query(
            args.question,
            args.top_k,
            show_sources=not args.no_sources,
            use_rerank=not args.no_rerank
        )

    elif args.docs:
        # 文檔檢索模式
        query_system.get_relevant_documents(args.docs, args.top_k)

    elif args.batch:
        # 批次查詢模式
        query_system.batch_query(args.batch, args.output)

    elif args.interactive:
        # 交互式模式
        query_system.interactive_mode()

    else:
        # 默認進入交互式模式
        print("未指定查詢模式，進入交互式模式...")
        query_system.interactive_mode()


if __name__ == "__main__":
    main()