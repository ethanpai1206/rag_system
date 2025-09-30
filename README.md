# RAG System - 檢索增強生成系統

基於 LlamaIndex、Milvus 和 OpenAI 的 RAG（Retrieval-Augmented Generation）系統，提供文檔智能問答服務。

## 🚀 功能特性

- **文檔處理**：支援 PDF 文件解析和文字處理
- **向量化存儲**：使用 Milvus 向量資料庫進行高效檢索
- **智能問答**：結合 OpenAI GPT 模型生成準確回答
- **本地查詢**：支援命令列和交互式查詢模式
- **模組化設計**：數據處理和查詢服務完全分離
- **批次處理**：支援大量文檔的批次入庫和查詢

## 📊 文檔索引流程

```mermaid
flowchart LR
    A[📄 PDF 讀取<br/>pdfminer] --> B[✂️ 語意切分<br/>SemanticSplitter]
    B --> C[🔢 向量化<br/>text-embedding-3-small<br/>1536 維]
    C --> D[💾 Milvus<br/>向量資料庫]

    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e9
```

| 步驟 | 說明 | 使用技術 |
|------|------|----------|
| 📄 讀取 | 提取 PDF 文字內容 | pdfminer |
| ✂️ 切分 | 智能語意切分，避免生硬斷句 | SemanticSplitterNodeParser |
| 🔢 向量化 | 文字轉 1536 維向量 | OpenAI text-embedding-3-small |
| 💾 存儲 | 向量索引建立與存儲 | Milvus VectorStore |

## 🔍 查詢問答流程

```mermaid
flowchart LR
    A[❓ 使用者提問] --> B[🔢 問題向量化<br/>text-embedding-3-small<br/>1536 維]
    B --> C[🔍 向量檢索<br/>Milvus<br/>Top-K 相關文檔]
    C --> D[🎯 重排序<br/>MxbaiRerankV2<br/>精準排序]
    D --> E[📝 組合 Prompt<br/>問題 + 檢索結果<br/>套入模板]
    E --> F[🤖 LLM 生成<br/>OpenAI GPT<br/>生成答案]
    F --> G[💬 返回結果]

    style A fill:#e8f5e9
    style B fill:#fff3e0
    style C fill:#e3f2fd
    style D fill:#f3e5f5
    style E fill:#fce4ec
    style F fill:#e1f5fe
    style G fill:#e8f5e9
```

| 步驟 | 說明 | 使用技術 |
|------|------|----------|
| ❓ 提問 | 使用者輸入問題 | 命令列 |
| 🔢 向量化 | 將問題轉換為 1536 維向量 | OpenAI text-embedding-3-small |
| 🔍 檢索 | 在向量資料庫中找出最相關的 Top-K 文檔 | Milvus 語意搜尋 |
| 🎯 重排序 | 使用 Rerank 模型精準排序檢索結果 | MxbaiRerankV2 |
| 📝 組合 | 將問題和檢索結果套入 Prompt 模板 | Custom PromptTemplate |
| 🤖 生成 | LLM 根據上下文生成答案 | OpenAI GPT-4 |
| 💬 返回 | 返回答案和來源資訊 | JSON / 命令列輸出 |

## 📁 專案結構

```
rag_system/
├── shared_config.py          # 共享配置
├── document_indexing.py      # 文檔索引模組
├── local_query.py            # 本地查詢工具
├── docker-compose-milvus.yml # Milvus 容器配置
├── requirements.txt          # Python 依賴
├── .env.example             # 環境變數範例
└── README.md                # 本文件
```

## 🛠️ 環境需求

### 方法一：本地安裝
- Python 3.10+
- Docker 和 Docker Compose
- OpenAI API 金鑰

### 方法二：Docker 開發環境（推薦）
- Docker 和 Docker Compose
- NVIDIA GPU（可選，用於 GPU 加速）
- OpenAI API 金鑰

## 📦 安裝步驟

### 方法一：本地安裝（手動配置）

#### 1. 克隆專案
```bash
git clone <your-repo-url>
cd rag_system
```

#### 2. 安裝 Python 依賴
```bash
pip install -r requirements.txt
```

#### 3. 設置環境變數
```bash
cp .env.example .env
# 編輯 .env 文件，填入你的 OpenAI API 金鑰
```

**.env 檔案內容：**
```env
OPENAI_API_KEY=your_openai_api_key_here
```

#### 4. 啟動 Milvus 向量資料庫
```bash
docker-compose -f docker-compose-milvus.yml up -d
```

等待所有服務啟動完成（大約 30-60 秒）。

#### 5. 驗證 Milvus 運行狀態
```bash
# 檢查容器狀態
docker-compose -f docker-compose-milvus.yml ps

# 檢查 Milvus 健康狀態
curl http://localhost:9091/healthz
```

---

### 方法二：Docker 完整開發環境（推薦）

此方法將整個開發環境打包到 Docker 容器中，包含所有依賴和配置。

#### 1. 克隆專案
```bash
git clone <your-repo-url>
cd rag_system
```

#### 2. 設置環境變數
```bash
cp .env.example .env
# 編輯 .env 文件，填入你的 OpenAI API 金鑰
```

#### 3. 啟動完整環境
```bash
# 啟動 Milvus 向量資料庫
docker-compose -f docker-compose-milvus.yml up -d

# 構建並啟動開發容器
docker-compose up -d --build
```

#### 4. 進入開發容器
```bash
docker exec -it rag_system bash
```

#### 5. 容器內已包含
- ✅ Python 3.10 + 所有依賴套件
- ✅ CUDA 12.8 + cuDNN（支援 GPU）
- ✅ Node.js 18 + Claude Code CLI
- ✅ 完整的開發工具鏈

#### 6. 在容器內執行指令
```bash
# 文檔索引
python document_indexing.py --pdf your_document.pdf

# 查詢問答
python local_query.py -q "你的問題"
```

#### 7. 停止環境
```bash
# 停止開發容器
docker-compose down

# 停止 Milvus（可選）
docker-compose -f docker-compose-milvus.yml down
```

## 🚀 使用指南

### 第一階段：文檔索引

#### 處理單個 PDF 文件（使用提供的測試檔案）
```bash
python document_indexing.py --pdf llama2.pdf
```

#### 批次處理目錄中的 PDF 文件
```bash
python document_indexing.py --directory ./documents --pattern "*.pdf"
```

#### 直接輸入文字資料
```bash
python document_indexing.py --text "這是第一段文字" "這是第二段文字"
```

#### 示例模式（測試用）
```bash
python document_indexing.py
```

#### 清空數據庫
```bash
python document_indexing.py --clear
```

### 第二階段：查詢服務

#### 本地查詢工具

```bash
# 交互式模式（預設）
python local_query.py

# 單次查詢（基於 llama2.pdf 的內容）
python local_query.py -q "Llama 2有多少参数"

# 預期回答：
# Llama 2 提供多个参数规模：7B、13B 和 70B（论文中还报告了34B版本，但未发布）。

# 檢索相關文檔
python local_query.py -d "Llama 2" -k 3

# 批次查詢
python local_query.py -b "Llama 2有多少参数" "Llama 2的训练数据是什么" -o results.json
```

## 💻 本地查詢工具詳細說明

### 交互式模式

```bash
python local_query.py -i
# 或直接執行
python local_query.py
```

進入交互式模式後，可使用以下命令：

- **直接輸入問題** - 獲得完整的問答回應
- **docs [問題]** - 僅檢索相關文檔，不生成回答
- **help** - 顯示命令說明
- **quit/exit/q** - 退出程序

### 命令列參數

```bash
# 單次查詢問題
python local_query.py -q "什麼是人工智慧？" -k 5

# 檢索相關文檔
python local_query.py -d "機器學習的應用" -k 3

# 批次查詢並保存結果
python local_query.py -b "AI是什麼？" "深度學習原理" -o output.json

# 不顯示來源信息
python local_query.py -q "什麼是NLP？" --no-sources
```

### 參數說明

- `-q, --question`：單次查詢問題
- `-d, --docs`：檢索相關文檔
- `-k, --top-k`：檢索文檔數量
- `-i, --interactive`：交互式模式
- `-b, --batch`：批次查詢問題列表
- `-o, --output`：輸出文件路徑
- `--no-sources`：不顯示來源信息

## 🐍 Python 程式範例

### 本地查詢系統調用

```python
from local_query import LocalQuerySystem

# 初始化查詢系統
query_system = LocalQuerySystem()

# 單次查詢
result = query_system.query("什麼是人工智慧？", top_k=3)
print(f"回答: {result['answer']}")

# 檢索相關文檔
docs = query_system.get_relevant_documents("機器學習", top_k=5)
for i, doc in enumerate(docs, 1):
    print(f"{i}. 分數: {doc['score']:.4f}")
    print(f"   內容: {doc['text'][:100]}...")

# 批次查詢
questions = ["什麼是AI？", "深度學習的原理？"]
results = query_system.batch_query(questions, "batch_results.json")
```

## ⚙️ 配置選項

在 `shared_config.py` 中可以調整以下設定：

```python
# OpenAI 配置
EMBEDDING_MODEL = "text-embedding-3-small"  # 嵌入模型
LLM_MODEL = "gpt-4"                         # 語言模型
LLM_TEMPERATURE = 0.1                       # 生成溫度

# Milvus 配置
MILVUS_HOST = "localhost"                   # Milvus 主機
MILVUS_PORT = 19530                        # Milvus 端口
MILVUS_COLLECTION_NAME = "rag_documents"   # 集合名稱

# 文檔處理配置
CHUNK_SIZE = 512                           # 分塊大小
CHUNK_OVERLAP = 50                         # 分塊重疊
SIMILARITY_TOP_K = 5                       # 預設檢索數量

# API 配置
API_HOST = "0.0.0.0"                      # API 監聽地址
API_PORT = 8000                           # API 端口
```

## 🔧 故障排除

### 常見問題

**1. OpenAI API 金鑰錯誤**
```
解決方案：檢查 .env 文件中的 OPENAI_API_KEY 是否正確設置
```

**2. Milvus 連接失敗**
```bash
# 檢查 Milvus 服務狀態
docker-compose -f docker-compose-milvus.yml ps
# 重啟 Milvus 服務
docker-compose -f docker-compose-milvus.yml restart
```

**3. PDF 解析失敗**
```
解決方案：確保 PDF 文件沒有密碼保護，且格式正確
```

**4. 記憶體不足**
```
解決方案：調整 CHUNK_SIZE 為更小的值，或分批處理文檔
```

### 日誌檢查

```bash
# 查看 Milvus 日誌
docker-compose -f docker-compose-milvus.yml logs standalone

# 查看 Python 應用日誌
python query_service.py  # 直接運行查看輸出
```

## 🔍 監控和維護

### Milvus Web UI
- 訪問：http://localhost:9001
- 用戶名：minioadmin
- 密碼：minioadmin

### 資料庫管理
```bash
# 停止服務
docker-compose -f docker-compose-milvus.yml down

# 清除資料（謹慎使用）
docker-compose -f docker-compose-milvus.yml down -v

# 重新啟動
docker-compose -f docker-compose-milvus.yml up -d
```

## 📈 性能優化

1. **調整 CHUNK_SIZE**：根據文檔特性調整分塊大小
2. **批次處理**：使用目錄批次模式處理大量文檔
3. **索引優化**：在 Milvus 中調整索引參數
4. **硬體配置**：為 Milvus 分配足夠的記憶體和 CPU


## 📄 授權條款

本專案採用 MIT 授權條款 - 查看 [LICENSE](LICENSE) 文件了解詳情。


## 🙏 致謝

- [LlamaIndex](https://www.llamaindex.ai/) - 提供強大的 RAG 框架
- [Milvus](https://milvus.io/) - 高性能向量資料庫
- [OpenAI](https://openai.com/) - 先進的語言模型服務
- [FastAPI](https://fastapi.tiangolo.com/) - 現代化的 Web 框架