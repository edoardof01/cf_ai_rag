# PROMPTS.md — AI Prompts Used During Development

This document catalogs the key AI prompts used during the development of the **cf_ai_rag** project.  
Development was carried out using **Google Gemini 3.1 Pro** and **Anthropic Claude Opus 4.6** through the [Antigravity](https://github.com/google-deepmind/antigravity) AI coding assistant.

> **Note:** Prompts are grouped by development phase. Minor follow-up messages (e.g., "Continue", "go on") are omitted for brevity.

---

## Table of Contents

- [Phase 1 — Project Scaffolding & Core Pipeline](#phase-1--project-scaffolding--core-pipeline)
- [Phase 2 — Kubernetes Infrastructure](#phase-2--kubernetes-infrastructure)
- [Phase 3 — FastAPI REST Microservice](#phase-3--fastapi-rest-microservice)
- [Phase 4 — Multi-Strategy Conversational Memory](#phase-4--multi-strategy-conversational-memory)
- [Phase 5 — Hybrid Retrieval & Theoretical Extensions](#phase-5--hybrid-retrieval--theoretical-extensions)
- [Phase 6 — Deployment Automation & Reliability](#phase-6--deployment-automation--reliability)
- [Phase 7 — Technical Report](#phase-7--technical-report)
- [Phase 8 — Repository Finalization](#phase-8--repository-finalization)

---

## Phase 1 — Project Scaffolding & Core Pipeline

### Prompt 1.1 — Project Skeleton Generation
> **Model:** Gemini 3.1 Pro  
> **Context:** Empty project directory, architecture already designed on paper

```
Generate the Python module skeleton for a RAG system with the following separation of concerns:
pdf_loader (PyMuPDF extraction), chunker (fixed-size with overlap), embedder (sentence-transformers wrapper),
vector_store (Milvus client), llm_client (Ollama REST), rag_pipeline (orchestrator), main (CLI entry point).
Each module should have full docstrings and type hints. Use dataclasses for data transfer objects.
```

### Prompt 1.2 — Embedding Normalization Strategy
> **Model:** Gemini 3.1 Pro

```
I want to use Inner Product as the similarity metric in Milvus instead of L2 distance.
This requires L2-normalized embeddings so that IP becomes equivalent to cosine similarity.
Verify that sentence-transformers' encode() with normalize_embeddings=True produces
unit-norm vectors and confirm the mathematical equivalence: IP(u,v) = cos(u,v) when ||u||=||v||=1.
```

---

## Phase 2 — Kubernetes Infrastructure

### Prompt 2.1 — K8s Manifest Generation
> **Model:** Gemini 3.1 Pro  
> **Context:** Working local system, migrating from Docker Compose to Kubernetes

```
Generate Kubernetes YAML manifests for the following components on Minikube:
- Milvus standalone with etcd (meta-storage) and MinIO (object storage), each with PVCs
- Redis single-instance deployment with a ClusterIP service
- Ollama deployment exposing port 11434
- ConfigMap-based environment injection instead of .env files
The Milvus Dockerfile should pre-bake the embedding model to avoid runtime downloads.
```

### Prompt 2.2 — IVF_FLAT Index Tuning
> **Model:** Gemini 3.1 Pro

```
For the Milvus collection, I'm choosing IVF_FLAT over HNSW.
With a corpus under 100k vectors, what's the optimal nlist parameter?
Also, what nprobe value gives the best recall/latency tradeoff at search time?
I want at least 95% recall compared to brute-force.
```

> Used to calibrate `nlist=128` and `nprobe=16` in `vector_store.py`.

---

## Phase 3 — FastAPI REST Microservice

### Prompt 3.1 — Lifespan Pattern for Model Initialization
> **Model:** Gemini 3.1 Pro  
> **Context:** Need to load 3 AI models (embedder, cross-encoder, LLM client) exactly once at startup

```
Implement a FastAPI lifespan context manager that initializes the embedding model,
cross-encoder reranker, and Ollama client once at startup, making them available
as module-level singletons. The lifespan must also establish connections to Milvus and Redis,
and rebuild the BM25 index from Milvus if the Redis cache is cold.
Include a /health endpoint suitable for Kubernetes liveness probes.
```

### Prompt 3.2 — Code Review: In-Memory PDF Processing
> **Model:** Gemini 3.1 Pro

```
I've implemented the /upload-pdf endpoint with zero disk I/O — reading bytes directly
from FastAPI's UploadFile, opening them with fitz.open(stream=pdf_bytes, filetype="pdf"),
then chunking, embedding, and inserting into both Milvus and BM25 in a single request.
Review my implementation for edge cases I might have missed:
- What happens with encrypted/password-protected PDFs?
- Are there memory issues with very large files (100+ MB)?
- Is the error handling correct for non-extractable scanned PDFs?
```

---

## Phase 4 — Multi-Strategy Conversational Memory

### Prompt 4.1 — Review: Memory Architecture & φ Function Selection
> **Model:** Gemini 3.1 Pro  
> **Context:** Implemented triple-strategy memory in memory.py, need review on φ function design

```
I've implemented a conversational memory module (memory.py) with three strategies:
1. KV: Redis List with FIFO sliding window (ltrim to W elements)
2. Summary: Redis String with cumulative LLM-generated summaries every N turns
3. Vector Memory: Milvus collection storing rewritten turn embeddings

I'm unsure about my query extension function φ(M_{t-1}). I've drafted three candidates:
  - φ_text(M) = "[Summary: S_{t-1}]" — simple but loses episodic detail
  - φ_vec(M) = "[Related turn: VM.search(q_t, 1)]" — precise but no global context
  - φ_hybrid(M) = φ_text ⊕ φ_vec — combines both but increases token usage

Review my compute_phi() implementation and advise:
- Is φ_hybrid worth the extra tokens, or does the summary alone suffice?
- My dual retrieval C_t = Retrieve_KB(q'_t) ∪ Retrieve_VM(q_t) uses q'_t for KB
  but q_t for VM — is this asymmetry correct, or should VM also get the augmented query?
```

> After discussion, `φ_hybrid` was confirmed as optimal. The asymmetry was validated: VM should use the original `q_t` to avoid feedback loops where past context biases future memory retrieval.

### Prompt 4.2 — Query Rewriting for Vector Memory
> **Model:** Gemini 3.1 Pro

```
Before embedding conversation turns into the Vector Memory, I want to apply LLM-based
query rewriting. Instead of storing raw "Q: ... A: ..." pairs, the LLM should distill
each turn into a single standalone sentence that captures the core semantic content.
This improves retrieval quality because the rewritten text is decontextualized and
more aligned with future search queries. Include a fallback to raw format if the LLM fails.
```

---

## Phase 5 — Hybrid Retrieval & Theoretical Extensions

### Prompt 5.1 — Debug: RRF Fusion Producing Unexpected Rankings
> **Model:** Claude Opus 4.6

```
I've implemented BM25 hybrid retrieval with Reciprocal Rank Fusion in rag_pipeline.py.
My _reciprocal_rank_fusion() method merges vector and BM25 results using:
  score(d) = Σ 1/(k + rank_i) for each retriever i that found document d

But I'm seeing unexpected behavior: documents that appear in BOTH retrievers sometimes
rank lower than documents found by only one. I suspect my deduplication logic is wrong —
I'm using the chunk text as the unique key, but text might differ slightly between
the two stores (whitespace, encoding).

Also: is k=60 the right constant? I've seen papers suggest k=1 for small result sets.
Review my implementation and suggest fixes.
```

> The AI identified that the deduplication was correct (text identity is preserved since both stores index the same preprocessed strings), but the ranking issue was caused by a missing accumulation step. The fix was confirmed and k=60 validated as the standard constant from the original RRF paper (Cormack et al., 2009).

### Prompt 5.2 — Theoretical Extensions Brainstorming
> **Model:** Claude Opus 4.6

```
Propose mathematical/theoretical extensions to improve this RAG system beyond basic
retrieval. I'm specifically interested in:
- Information-theoretic approaches (entropy, mutual information)
- Optimization formulations (the token budget as a knapsack problem)
- Submodular diversification (MMR-style chunk selection)
- Calibration and uncertainty quantification for generated answers
Frontend/UI suggestions are out of scope.
```

> The AI produced 7 ranked proposals. From this analysis, the BM25 hybrid retrieval, RRF fusion, and Cross-Encoder re-ranking pipeline was designed and implemented.

---

## Phase 6 — Deployment Automation & Reliability

### Prompt 6.1 — Bootstrap Script Design
> **Model:** Gemini 3.1 Pro  
> **Context:** Manual deployment involved 10+ sequential commands with timing dependencies

```
I've designed a 6-phase bootstrap sequence for start.sh:
1. Minikube startup with resource allocation (6GB RAM, 4 CPUs)
2. kubectl apply for all K8s manifests
3. Pod readiness polling (kubectl wait --for=condition=available)
4. Port-forward setup with TCP connectivity probes
5. LLM model pull via Ollama REST API (curl to /api/pull)
6. FastAPI server start + automatic PDF bulk upload

Generate the bash implementation with colored output, error handling at each phase,
and retry loops for the connectivity probes (Milvus, Redis, Ollama).
```

### Prompt 6.2 — TCP vs gRPC Readiness Race Condition
> **Model:** Gemini 3.1 Pro  
> **Context:** Milvus TCP port 19530 opens before the internal gRPC service is initialized

```
There's a race condition in the bootstrap: the TCP probe on localhost:19530 succeeds
(port is open), but Milvus's internal gRPC service isn't yet initialized, causing
pymilvus connection failures. I need a dual-layer solution:
1. A post-TCP-probe delay in start.sh (heuristic guard)
2. Exponential backoff retry logic in the Python MilvusVectorStore constructor
Implement both with configurable max_retries and retry_delay parameters.
```

### Prompt 6.3 — Metadata-Aware Chunk Enrichment
> **Model:** Gemini 3.1 Pro

```
The PDF filenames in my corpus encode structured metadata
(e.g., "Acme_BackendEngineer_Milan_Hybrid.pdf").
I want this metadata to be searchable by both the vector and BM25 retrievers.
Strategy: prefix each chunk's text with the filename before embedding and indexing:
"[Acme_BackendEngineer_Milan_Hybrid.pdf]\nActual chunk text..."
This way the metadata tokens participate in both dense and sparse retrieval
without requiring a separate metadata schema.
```

---

## Phase 7 — Technical Report

### Prompt 7.1 — LaTeX Report Section Generation
> **Model:** Claude Opus 4.6  
> **Context:** Report skeleton with 4 existing sections, need 4 more with full mathematical formalization

```
Generate the following LaTeX sections for the technical report, with proper mathematical
notation (amsmath), code listings, and formal definitions:
1. Ingestion Pipeline — PDF extraction, fixed-size chunking formula with overlap
2. Embedding Theory — L2 normalization proof, IP-cosine equivalence, IVF_FLAT indexing
3. Prompt Engineering — Template structure, grounding strategies, temperature formalization
4. Deployment — K8s microservice topology, lifespan pattern, retry strategies
All sections must be in Italian and include references to specific source files.
```

---

## Phase 8 — Repository Finalization

### Prompt 8.1 — Submission Compliance
> **Model:** Claude Opus 4.6

```
Prepare the repository for submission:
- Rewrite README.md with comprehensive English documentation and clear running instructions
- Create PROMPTS.md documenting all AI prompts used during development
- Update .gitignore for clean submission (exclude build artifacts, LaTeX intermediaries, venvs)
- Repository must be prefixed with cf_ai_
```

---

## Summary of AI Tools Used

| Tool | Model(s) | Usage |
|------|----------|-------|
| **Antigravity** (AI coding assistant) | Gemini 3.1 Pro, Claude Opus 4.6 | Code generation, debugging, documentation |
| **Ollama** (local LLM, runtime component) | Qwen 2.5 7B | Response generation, memory summarization, query rewriting |
