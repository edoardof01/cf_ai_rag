"""
main.py — Entry point CLI per il sistema RAG.

Avvia un loop interattivo dove l'utente può fare domande
e ricevere risposte basate sui documenti PDF indicizzati.

Uso: python main.py
"""

import os
import sys

from dotenv import load_dotenv

from src.embedder import Embedder
from src.vector_store import MilvusVectorStore
from src.llm_client import OllamaClient
from src.rag_pipeline import RAGPipeline

load_dotenv()


def main():
    print("=" * 60)
    print("🤖 RAG System — Chat con i tuoi documenti")
    print("=" * 60)

    # Configurazione
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    collection_name = os.getenv("MILVUS_COLLECTION", "rag_documents")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", 384))
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    top_k = int(os.getenv("TOP_K", 5))

    # Inizializzazione componenti
    print("\n⏳ Inizializzazione componenti...\n")
    embedder = Embedder(model_name)
    vector_store = MilvusVectorStore(
        host=milvus_host, port=milvus_port,
        collection_name=collection_name, embedding_dim=embedding_dim,
    )
    llm_client = OllamaClient(base_url=ollama_url, model=ollama_model)

    # Verifica Ollama
    if not llm_client.is_available():
        print(f"❌ Ollama non raggiungibile su {ollama_url}")
        print("   Avvia Ollama con: ollama serve")
        sys.exit(1)

    print(f"\n📊 Documenti nel DB: {vector_store.count()} chunk")
    if vector_store.count() == 0:
        print("⚠️  Nessun documento indicizzato!")
        print("   Esegui prima: python -m scripts.index_documents")
        print()

    pipeline = RAGPipeline(embedder, vector_store, llm_client, top_k=top_k)

    # Loop interattivo
    print("\n💬 Fai una domanda (digita 'exit' per uscire):\n")
    while True:
        try:
            question = input("❓ > ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "esci", "done"):
                print("👋 Arrivederci!")
                break

            response = pipeline.query(question)
            print(response)

        except KeyboardInterrupt:
            print("\n👋 Arrivederci!")
            break


if __name__ == "__main__":
    main()
