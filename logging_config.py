import os
import logging
import json
from datetime import datetime
from typing import Dict, Any


class QueryLogger:
    """查詢日誌記錄器"""

    def __init__(self, logger_name: str = "LocalQuerySystem", log_dir: str = "logs", enable_console: bool = True):
        """
        初始化日誌記錄器

        Args:
            logger_name: logger 名稱
            log_dir: 日誌目錄
            enable_console: 是否啟用 console 輸出
        """
        self.logger_name = logger_name
        self.log_dir = log_dir
        self.enable_console = enable_console
        self.logger = None
        self._setup_logging()

    def _setup_logging(self):
        """設置 logging 配置"""
        # 建立 logs 目錄
        os.makedirs(self.log_dir, exist_ok=True)

        # 設置 logger
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)

        # 避免重複添加 handler
        if not self.logger.handlers:
            # 文件 handler
            log_file = os.path.join(
                self.log_dir,
                f"query_log_{datetime.now().strftime('%Y%m%d')}.log"
            )

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)

            # 設置格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_query_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        記錄查詢結果到 log 文件

        Args:
            result: 查詢結果字典

        Returns:
            格式化的日誌數據
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": result["question"],
            "answer": result["answer"],
            "processing_time": result["processing_time"],
            "sources": []
        }

        # 處理來源資訊
        for i, source in enumerate(result["sources"], 1):
            source_info = {
                "rank": i,
                "similarity_score": source["score"],
                "metadata": source["metadata"],
                "full_content": source["text"]  # 保存完整內容
            }
            log_data["sources"].append(source_info)

        # 記錄到檔案
        self.logger.info(f"Query Result: {json.dumps(log_data, ensure_ascii=False, indent=2)}")

        return log_data

    def log_info(self, message: str):
        """記錄一般資訊"""
        self.logger.info(message)

    def log_error(self, message: str):
        """記錄錯誤資訊"""
        self.logger.error(message)

    def log_warning(self, message: str):
        """記錄警告資訊"""
        self.logger.warning(message)


def get_query_logger(logger_name: str = "LocalQuerySystem", log_dir: str = "logs", enable_console: bool = True) -> QueryLogger:
    """
    獲取查詢日誌記錄器實例

    Args:
        logger_name: logger 名稱
        log_dir: 日誌目錄
        enable_console: 是否啟用 console 輸出

    Returns:
        QueryLogger 實例
    """
    return QueryLogger(logger_name, log_dir, enable_console)