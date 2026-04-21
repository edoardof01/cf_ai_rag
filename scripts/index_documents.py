"""
index_documents.py — Script per indicizzare i PDF nel database vettoriale.

Uso: python -m scripts.index_documents [--data-dir ./data] [--reset]
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Aggiungi la root del progetto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_loader import load_pdfs_from_directory
from src.chunker import chunk_pages
from src.embedder import Embedder
from src.vector_store import MilvusVectorStore

load_dotenv()


def main():
    # Se lanci python index_documents.py --data-dir ./pdf_segreti --reset, 
    # cambierà cartella e, grazie al --reset, distruggerà il database Milvus 
    # ripartendo da zero! Molto utile durante i test.
    parser = argparse.ArgumentParser(description="Indicizza i PDF nel database vettoriale")
    parser.add_argument("--data-dir", default="./data", help="Directory con i PDF")
    parser.add_argument("--reset", action="store_true", help="Elimina e ricrea la collection")
    args = parser.parse_args()

    # Configurazione da .env
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", 384))
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    collection_name = os.getenv("MILVUS_COLLECTION", "rag_documents")

    print("=" * 60)
    print("📚 RAG Document Indexer")
    print("=" * 60)

    # Step 1: Caricamento PDF
    print(f"\n📁 Caricamento PDF da: {args.data_dir}")
    pages = load_pdfs_from_directory(args.data_dir)
    if not pages:
        print("❌ Nessuna pagina da indicizzare. Aggiungi PDF nella cartella data/")
        return

    # Step 2: Chunking
    print(f"\n✂️  Chunking (size={chunk_size}, overlap={chunk_overlap})")
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"   → {len(chunks)} chunk generati")

    # Step 3: Embedding
    print(f"\n🧠 Generazione embedding con: {model_name}")
    embedder = Embedder(model_name)
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)

    # Step 4: Inserimento in Milvus
    print(f"\n💾 Inserimento in Milvus...")
    store = MilvusVectorStore(
        host=milvus_host, port=milvus_port,
        collection_name=collection_name, embedding_dim=embedding_dim,
    )
    if args.reset:
        store.drop_collection()
        store = MilvusVectorStore(
            host=milvus_host, port=milvus_port,
            collection_name=collection_name, embedding_dim=embedding_dim,
        )

    source_files = [c.source_file for c in chunks]
    page_numbers = [c.page_number for c in chunks]
    store.insert(texts, embeddings, source_files, page_numbers)

    print(f"\n{'=' * 60}")
    print(f"✅ Indicizzazione completata!")
    print(f"   Chunk totali nel DB: {store.count()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
