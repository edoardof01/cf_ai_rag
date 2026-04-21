"""
api.py — Endpoint REST con FastAPI per il sistema RAG.

Avvia l'applicazione in locale con:
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

import redis
import fitz  # PyMuPDF — usato per leggere il PDF direttamente dai byte in memoria
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

from src.embedder import Embedder
from src.vector_store import MilvusVectorStore
from src.bm25_store import BM25Store
from src.reranker import CrossEncoderReranker
from src.llm_client import OllamaClient
from src.rag_pipeline import RAGPipeline
from src.memory import ConversationMemory
from src.pdf_loader import PageContent
from src.chunker import chunk_pages

load_dotenv()


# Variabili globali inizializzate una sola volta nel lifespan
pipeline: RAGPipeline | None = None
embedder: Embedder | None = None
vector_store: MilvusVectorStore | None = None
bm25_store: BM25Store | None = None
reranker: CrossEncoderReranker | None = None
llm_client_global: OllamaClient | None = None
redis_client: redis.Redis | None = None


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"  # ID della sessione conversazionale

class QueryResponse(BaseModel):
    question: str
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Funzione eseguita all'avvio e allo spegnimento del server.
    Inizializza i modelli AI e la connessione al DB una sola volta.
    """
    global pipeline, embedder, vector_store, bm25_store, reranker, llm_client_global, redis_client

    print("\n⏳ [FastAPI Lifespan] Inizializzazione componenti RAG...\n")
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    collection_name = os.getenv("MILVUS_COLLECTION", "rag_documents")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", 384))
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    top_k = int(os.getenv("TOP_K", 10))
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    # Connessione Redis per le sessioni conversazionali e BM25
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
    print(f"🔗 [Redis] Connesso a {redis_host}:{redis_port}")

    embedder = Embedder(model_name)
    vector_store = MilvusVectorStore(
        host=milvus_host, port=milvus_port,
        collection_name=collection_name, embedding_dim=embedding_dim,
    )
    llm_client_global = OllamaClient(base_url=ollama_url, model=ollama_model)

    if not llm_client_global.is_available():
        print(f"⚠️  [Warning] Ollama non raggiungibile su {ollama_url}")

    # Inizializzazione Indice BM25 (Ibrido)
    bm25_store = BM25Store()
    if not bm25_store.load_from_redis(redis_client):
        print("Costruzione indice BM25 da zero usando Milvus...")
        texts, metadata = vector_store.get_all_texts()
        bm25_store.build_index(texts, metadata)
        bm25_store.save_to_redis(redis_client)

    # Inizializzazione Re-Ranker (Cross-Encoder)
    reranker = CrossEncoderReranker(model_name=reranker_model)
    
    pipeline = RAGPipeline(embedder, vector_store, bm25_store, reranker, llm_client_global, top_k=top_k)
    print("\n✅ [FastAPI Lifespan] RAG Pipeline pronta all'uso!\n")
    
    yield  # Il server gira qui
    
    print("\n🛑 [FastAPI Lifespan] Spegnimento server. Pulizia risorse...\n")


# Inizializzazione app FastAPI
app = FastAPI(
    title="RAG API",
    description="API REST per interrogare il database vettoriale sui documenti PDF.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """
    Liveness probe per Kubernetes. Ritorna 200 OK se l'app è viva.
    """
    return {"status": "ok", "message": "RAG API is running"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Endpoint principale per fare domande al sistema RAG.
    Riceve un JSON con "question", "session_id" e restituisce la "answer".
    La memoria conversazionale viene gestita automaticamente per sessione.
    Lo stato della sessione è persistente su Redis + Milvus.
    """
    if not pipeline or not embedder or not llm_client_global or not redis_client:
        raise HTTPException(status_code=500, detail="RAG Pipeline non inizializzata")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota")

    # Crea l'oggetto memoria puntando alla sessione su Redis + Milvus.
    # Non è un dizionario in-memory: lo stato vive nei database esterni,
    # quindi più Pod possono condividere la stessa sessione.
    memory = ConversationMemory(
        session_id=request.session_id,
        redis_client=redis_client,
        llm_client=llm_client_global,
        embedder=embedder,
    )

    try:
        answer = pipeline.query(request.question, memory=memory)
        return QueryResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """
    Cancella una sessione conversazionale, azzerando la memoria su Redis e Milvus.
    """
    if not redis_client or not embedder or not llm_client_global:
        raise HTTPException(status_code=500, detail="Componenti non inizializzati")

    memory = ConversationMemory(
        session_id=session_id,
        redis_client=redis_client,
        llm_client=llm_client_global,
        embedder=embedder,
    )

    if memory.is_empty():
        raise HTTPException(status_code=404, detail=f"Sessione '{session_id}' non trovata")

    memory.clear()
    return {"status": "ok", "message": f"Sessione '{session_id}' eliminata"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint per caricare e indicizzare un file PDF.
    Riceve il file via HTTP multipart upload, estrae il testo,
    genera gli embedding e li salva in Milvus.
    Tutto avviene in memoria, senza scrivere nulla su disco.
    """
    if not embedder or not vector_store or not bm25_store or not redis_client:
        raise HTTPException(status_code=500, detail="Componenti RAG non inizializzati")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Il file deve essere un PDF")

    try:
        # 1. Leggi i byte del PDF dalla richiesta HTTP (in memoria, niente disco)
        pdf_bytes = await file.read()

        # 2. Apri il PDF direttamente dai byte con PyMuPDF
        pages: list[PageContent] = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append(PageContent(
                        text=text.strip(),
                        page_number=page_num,
                        source_file=file.filename,
                    ))

        if not pages:
            raise HTTPException(status_code=400, detail="Il PDF non contiene testo estraibile")

        # 3. Chunking
        chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 4. Embedding
        # Includiamo il nome del file nel testo indicizzato: contiene info
        # preziose (azienda, ruolo, città, "Remote") altrimenti invisibili alla ricerca.
        texts = [f"[{c.source_file}]\n{c.text}" for c in chunks]
        embeddings = embedder.embed_texts(texts)

        # 5. Inserimento in Milvus (Vector Search)
        source_files = [c.source_file for c in chunks]
        page_numbers = [c.page_number for c in chunks]
        vector_store.insert(texts, embeddings, source_files, page_numbers)

        # 6. Aggiornamento in BM25 (Lexical Search) e persistenza su Redis
        metadata = [{"source_file": sf, "page_number": pn} for sf, pn in zip(source_files, page_numbers)]
        if bm25_store.is_empty():
            bm25_store.build_index(texts, metadata)
        else:
            bm25_store.add_documents(texts, metadata)
        bm25_store.save_to_redis(redis_client)

        return {
            "status": "ok",
            "filename": file.filename,
            "pages_extracted": len(pages),
            "chunks_indexed": len(chunks),
        }

    except HTTPException:
        raise  # Rilancia le HTTPException già formattate
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'indicizzazione: {str(e)}")
