# 🤖 cf_ai_rag — Advanced Hybrid RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** system built as a microservice architecture on **Kubernetes (Minikube)**. The system queries a corpus of PDF documents using advanced hybrid retrieval, cross-encoder re-ranking, and multi-strategy conversational memory.

The entire stack is **fully self-hosted** — no paid external APIs required.

---

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [One-Command Startup](#one-command-startup)
  - [Stopping the System](#stopping-the-system)
- [How to Try It Out](#-how-to-try-it-out)
- [How It Works](#-how-it-works)
  - [Hybrid Retrieval](#1-hybrid-retrieval-semantic--lexical-bm25)
  - [Reciprocal Rank Fusion & Re-Ranking](#2-reciprocal-rank-fusion-rrf--cross-encoder-re-ranking)
  - [Multi-Strategy Conversational Memory](#3-multi-strategy-conversational-memory)
  - [Query Augmentation / Rewriting](#4-query-augmentation--rewriting)
  - [Token Budget Management](#5-token-budget-management)
  - [Metadata-Aware Chunking](#6-metadata-aware-chunking)
- [API Reference](#-api-reference)
- [Technical Report](#-technical-report)
- [AI-Assisted Development](#-ai-assisted-development)

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kubernetes (Minikube)                    │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │  Redis    │   │    Milvus    │   │       Ollama           │  │
│  │  :6379    │   │    :19530    │   │       :11434           │  │
│  │          │   │              │   │                        │  │
│  │ • BM25    │   │ • etcd       │   │ • Qwen 2.5 (7B)       │  │
│  │   Index   │   │ • MinIO      │   │   LLM inference       │  │
│  │ • Session │   │ • Standalone │   │                        │  │
│  │   Memory  │   │   VectorDB   │   │                        │  │
│  └─────┬─────┘   └──────┬───────┘   └────────────┬───────────┘  │
│        │                │                        │              │
│        └────────────────┼────────────────────────┘              │
│                         │                                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │ port-forward
                          │
              ┌───────────┴───────────┐
              │   FastAPI Server      │
              │   localhost:8000      │
              │                       │
              │ • /ask    (RAG query) │
              │ • /upload-pdf         │
              │ • /health             │
              │ • /docs   (Swagger)   │
              └───────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Retrieval** | Combines dense semantic search (Milvus) with sparse lexical search (BM25/Redis) |
| **Reciprocal Rank Fusion** | Merges results from both retrievers using RRF scoring |
| **Cross-Encoder Re-Ranking** | Final relevance scoring via multilingual Cross-Encoder (RoBERTa-based) |
| **Multi-Strategy Memory** | Sliding window (KV), cumulative summary (LLM-generated), and vector memory (semantic) |
| **Query Rewriting** | LLM-powered contextual query expansion using conversation history |
| **Token Budget Management** | Dynamic context trimming to stay within the LLM's context window |
| **Metadata-Aware Chunking** | File names (containing role, company, location) are embedded into every chunk |
| **One-Command Deploy** | Fully automated startup script handles Minikube, K8s manifests, model download, and PDF indexing |
| **Self-Hosted** | Runs entirely locally — no OpenAI, no cloud APIs, no costs |

---

## 🛠 Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Backend** | FastAPI + Uvicorn | REST API server |
| **Vector Database** | Milvus 2.4 (standalone) | Dense semantic search (IVF_FLAT index, Inner Product) |
| **Cache / Memory** | Redis | BM25 index persistence, session memory (KV + Summary) |
| **LLM** | Ollama + Qwen 2.5 (7B) | Response generation, summary updates, query rewriting |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) | 384-dim normalized embeddings |
| **Re-Ranking** | Cross-Encoder (`mmarco-mMiniLMv2-L12-H384-v1`) | Multilingual cross-encoder for pair-wise relevance |
| **Lexical Search** | `rank-bm25` (BM25Okapi) | Term-frequency-based document retrieval |
| **PDF Extraction** | PyMuPDF (fitz) | Text extraction from PDF documents |
| **Orchestration** | Kubernetes (Minikube) | Container orchestration with PVCs for data persistence |
| **Containerization** | Docker | Application packaging with pre-baked embedding model |

---

## 📁 Project Structure

```
cf_ai_rag/
├── src/                          # Core application source code
│   ├── api.py                    # FastAPI REST endpoints & lifespan initialization
│   ├── rag_pipeline.py           # Main RAG orchestrator (retrieval → generation)
│   ├── memory.py                 # Multi-strategy conversational memory (KV/Summary/Vector)
│   ├── embedder.py               # Sentence-Transformers embedding wrapper
│   ├── vector_store.py           # Milvus client for dense vector search
│   ├── bm25_store.py             # BM25 lexical search with Redis persistence
│   ├── reranker.py               # Cross-Encoder re-ranking module
│   ├── chunker.py                # Fixed-size text chunking with overlap
│   ├── pdf_loader.py             # PDF text extraction via PyMuPDF
│   └── llm_client.py             # Ollama REST API client (streaming support)
│
├── k8s/                          # Kubernetes manifests
│   ├── milvus.yaml               # Milvus standalone + etcd + MinIO (with PVCs)
│   ├── redis.yaml                # Redis deployment + service
│   ├── ollama.yaml               # Ollama deployment + service
│   ├── rag-api.yaml              # RAG API deployment + service + ingress
│   └── rag-bot.yaml              # CLI Pod for interactive testing
│
├── scripts/                      # Automation scripts
│   ├── start.sh                  # One-command full system bootstrap
│   ├── stop.sh                   # Graceful shutdown (port-forwards + FastAPI)
│   └── index_documents.py        # Batch PDF indexing CLI tool
│
├── docs/                         # PDF documents corpus (add your own PDFs here)
│   ├── Careers/                  # Example: job posting PDFs
│   └── Trend/                    # Example: industry trend PDFs
│
├── report/                       # LaTeX technical report
│   ├── main.tex                  # Report entry point
│   └── sections/                 # Individual report sections
│
├── Dockerfile                    # Docker image with pre-baked embedding model
├── requirements.txt              # Python dependencies
├── main.py                       # CLI entry point (alternative to API)
├── PROMPTS.md                    # AI prompts used during development
└── README.md                     # This file
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| **minikube** | v1.30+ | [Install guide](https://minikube.sigs.k8s.io/docs/start/) |
| **kubectl** | v1.28+ | Usually bundled with minikube |
| **Python** | 3.12+ | For the local FastAPI server and virtual environment |
| **Docker** | 20.10+ | Required by minikube (docker driver) |

> **💡 Resource Note:** The system allocates 6 GB RAM and 4 CPUs to Minikube. The Qwen 2.5 7B model requires ~4.7 GB of disk space (downloaded automatically on first run).

### One-Command Startup

```bash
# 1. Clone the repository
git clone https://github.com/edoardof01/cf_ai_rag.git
cd cf_ai_rag

# 2. Create a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Add your PDF documents to the docs/ folder
cp /path/to/your/documents/*.pdf docs/

# 4. Launch everything with a single command
./scripts/start.sh
```

The `start.sh` script automatically:

1. ✅ Starts Minikube (6 GB RAM, 4 CPUs)
2. ✅ Deploys all Kubernetes manifests (Redis, Milvus + etcd + MinIO, Ollama)
3. ✅ Waits for all pods to be healthy
4. ✅ Sets up port-forwarding (Milvus:19530, Redis:6379, Ollama:11434)
5. ✅ Verifies backend connectivity with retries
6. ✅ Downloads the Qwen 2.5 7B model into Ollama (first run only, ~4.7 GB)
7. ✅ Starts the FastAPI server locally on port 8000
8. ✅ Automatically uploads and indexes all PDFs in `docs/`

### Stopping the System

```bash
# Stop FastAPI + port-forwards (keeps Minikube data)
./scripts/stop.sh

# To completely destroy the cluster
minikube delete
```

---

## 🎮 How to Try It Out

Once `start.sh` completes, the system is fully operational. Here are several ways to interact with it:

### Option 1: Swagger UI (Recommended for Exploration)

Open your browser at **http://localhost:8000/docs** — this provides an interactive Swagger UI where you can test all endpoints visually.

### Option 2: cURL Commands

**Ask a question (RAG query with conversational memory):**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What companies offer remote positions?", "session_id": "demo"}' | jq
```

**Upload a new PDF document:**
```bash
curl -X POST http://localhost:8000/upload-pdf \
  -F "file=@path/to/your/document.pdf"
```

**Check system health:**
```bash
curl http://localhost:8000/health
```

**Delete a conversation session:**
```bash
curl -X DELETE http://localhost:8000/session/demo
```

### Option 3: CLI Interactive Mode

```bash
source .venv/bin/activate
python main.py
```

### Example Conversation

```
❓ > What companies are hiring for backend roles?

📚 Found 5 KB chunks (Hybrid RRF) + 0 memory turns (~2340 estimated tokens). Generating...

Based on the indexed documents, the following companies are hiring for backend roles:

1. **Acme Corp** (Milan, Hybrid) — Senior Backend Engineer position requiring
   Python and Kubernetes experience. [Source: Acme_Backend_Milan_Hybrid.pdf, p.1]

2. **TechStartup** (Remote) — Backend Developer with focus on microservices...
   [Source: TechStartup_Backend_Remote.pdf, p.2]

❓ > Which of these offer remote work?

📚 Found 5 KB chunks (Hybrid RRF) + 2 memory turns (~3100 estimated tokens). Generating...

From the companies discussed earlier, **TechStartup** offers fully remote positions...
```

> **Note:** The second question demonstrates the memory system — it understands "these" refers to the companies from the previous answer, even though the raw query wouldn't match any documents on its own.

---

## 🔬 How It Works

### 1. Hybrid Retrieval (Semantic + Lexical BM25)

Pure vector search struggles with exact keyword matches (names, acronyms, IDs). This system implements **dual retrieval**:

- **Dense Semantic Search** → Chunks are embedded with `all-MiniLM-L6-v2` and stored in Milvus. Excels at capturing the general meaning of the query.
- **Sparse Lexical Search** → BM25Okapi implementation backed by Redis. Excels at exact keyword matching (proper nouns, technical terms).

### 2. Reciprocal Rank Fusion (RRF) & Cross-Encoder Re-Ranking

The dual retrieval results are combined through a two-stage process:

1. **RRF Fusion:** Results from Milvus and BM25 are merged using `score = Σ 1/(k + rank)`. Documents found by both retrievers receive a significantly higher score.
2. **Cross-Encoder Re-Ranking:** The top-N RRF candidates pass through a multilingual Cross-Encoder that evaluates actual `(Query ↔ Document)` pair-wise relevance, producing the final chunk selection.

### 3. Multi-Strategy Conversational Memory

The system implements three complementary memory strategies persisted on Redis + Milvus:

| Strategy | Storage | Purpose |
|----------|---------|---------|
| **Sliding Window (KV)** | Redis List | Last W turns for immediate context |
| **Cumulative Summary** | Redis String | LLM-generated summary updated every N turns |
| **Vector Memory (VM)** | Milvus Collection | Embeds rewritten turns for semantic retrieval of facts discussed 100+ turns ago |

### 4. Query Augmentation / Rewriting

When the user asks *"What are the requirements for this position?"*, a direct search would fail. The system performs **Query Rewriting**: the LLM analyzes the memory and rewrites the query as a standalone question (e.g., *"What are the requirements for the Senior Backend Engineer position at Acme Corp?"*) **before** querying the database.

Formally: `q'_t = q_t ⊕ φ(M_{t-1})` where `φ` combines Summary + Vector Memory retrieval.

### 5. Token Budget Management

The system enforces `|C_t| + |KV_{t-1}| + |q'_t| ≤ L_max` by dynamically trimming context. Trimming priority (least to most important):
1. KB chunks with lowest relevance score
2. Oldest KV history turns
3. Truncation of remaining KV text

### 6. Metadata-Aware Chunking

During ingestion, the filename (containing metadata like role, company, location, remote/hybrid) is prefixed to every chunk text: `[filename.pdf]\nText...`. This allows both the vector search and BM25 to find documents based on structural metadata, not just content.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Ask a question (RAG query with memory) |
| `POST` | `/upload-pdf` | Upload and index a new PDF document |
| `DELETE` | `/session/{id}` | Delete a conversation session |
| `GET` | `/health` | Liveness probe (Kubernetes-ready) |
| `GET` | `/docs` | Interactive Swagger UI |

### POST `/ask`

**Request:**
```json
{
  "question": "Your question here",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "question": "Your question here",
  "answer": "Generated answer with source citations..."
}
```

### POST `/upload-pdf`

**Request:** Multipart form upload with a PDF file.

**Response:**
```json
{
  "status": "ok",
  "filename": "document.pdf",
  "pages_extracted": 12,
  "chunks_indexed": 45
}
```

---

## 📄 Technical Report

A comprehensive LaTeX report is included in the `report/` directory, formalizing every component with mathematical notation:

- **Ingestion Pipeline** — PDF extraction, fixed-size chunking with overlap
- **Embedding Theory** — L2 normalization, IP/cosine equivalence, IVF_FLAT indexing
- **Conversational Memory** — KV/Summary/Vector memory formalism, φ function
- **Hybrid Retrieval** — RRF fusion formula, Cross-Encoder scoring
- **BM25 Theory** — Full BM25 scoring derivation with TF-IDF
- **Prompt Engineering** — Template structure, grounding, temperature effects
- **Kubernetes Deployment** — Microservice topology, provisioning automation

To compile: `cd report && pdflatex main.tex` (or upload to [Overleaf](https://overleaf.com)).

---

## 🤖 AI-Assisted Development

This project was developed with significant AI assistance using **Google Gemini** and **Anthropic Claude** through the Antigravity coding assistant. All prompts used during development are documented in [`PROMPTS.md`](PROMPTS.md).

---

## 📝 License

This project was developed as a coursework submission. All rights reserved.
