"""
vector_store.py — Client per il database vettoriale Milvus.

Milvus è un database ottimizzato per la ricerca di vettori per similarità.
"""

from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility,
)
import numpy as np


class MilvusVectorStore:
    """Client per interagire con Milvus per operazioni CRUD sui vettori."""

    def __init__(self, host="localhost", port="19530",
                 collection_name="rag_documents", embedding_dim=384,
                 max_retries=10, retry_delay=5):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        print(f"🔌 Connessione a Milvus ({host}:{port})...")

        # Retry con backoff: Milvus impiega tempo ad inizializzare il servizio gRPC
        # anche dopo che la porta TCP è già aperta.
        for attempt in range(1, max_retries + 1):
            try:
                connections.connect("default", host=host, port=port, timeout=10)
                print("✅ Connesso a Milvus")
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"❌ Impossibile connettersi a Milvus dopo {max_retries} tentativi")
                    raise
                print(f"⏳ Milvus non ancora pronto (tentativo {attempt}/{max_retries}), riprovo tra {retry_delay}s...")
                import time
                time.sleep(retry_delay)

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            print(f"📂 Collection '{self.collection_name}' trovata")
            col = Collection(self.collection_name)
            col.load()
            return col

        print(f"🆕 Creazione collection '{self.collection_name}'...")
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("source_file", DataType.VARCHAR, max_length=512),
            FieldSchema("page_number", DataType.INT64),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]
        schema = CollectionSchema(fields, description="RAG document chunks")
        col = Collection(self.collection_name, schema)

        # IVF_FLAT index: bilancia velocità e accuratezza
        col.create_index("embedding", {
            "metric_type": "IP",       # Inner Product (cosine per vettori normalizzati)
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        })
        print(f"✅ Collection creata con indice IVF_FLAT")
        col.load()
        return col

    def insert(self, texts, embeddings, source_files, page_numbers) -> int:
        """Inserisce chunk con embedding in Milvus."""
        data = [texts, source_files, page_numbers, embeddings.tolist()] # Seguiamo la struttura dello schema
        self.collection.insert(data)
        self.collection.flush()
        print(f"💾 Inseriti {len(texts)} chunk in Milvus")
        return len(texts)

    def search(self, query_embedding: np.ndarray, top_k=5) -> list[dict]:
        """Cerca i chunk più simili a un vettore query."""
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["text", "source_file", "page_number"],
        )
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "text": hit.entity.get("text"),
                    "source_file": hit.entity.get("source_file"),
                    "page_number": hit.entity.get("page_number"),
                    "score": hit.score,
                })
        return formatted

    def count(self) -> int:
        return self.collection.num_entities

    def get_all_texts(self) -> tuple[list[str], list[dict]]:
        """
        Recupera tutti i testi e metadati presenti nella collection.
        Viene usato all'avvio per ricostruire l'indice BM25.
        
        Returns:
            Una tupla (texts, metadata) dove metadata contiene source_file e page_number.
        """
        # Eseguiamo una query per prendere tutte le entità.
        # "id >= 0" è una tautologia per prendere tutto (il campo primario è id)
        results = self.collection.query(
            expr="id >= 0",
            output_fields=["text", "source_file", "page_number"],
            limit=16384  # Limite per non intasare la RAM, in prod si usa un iterator
        )
        
        texts = []
        metadata = []
        for res in results:
            texts.append(res.get("text"))
            metadata.append({
                "source_file": res.get("source_file"),
                "page_number": res.get("page_number")
            })
            
        return texts, metadata

    def drop_collection(self):
        utility.drop_collection(self.collection_name)
        print(f"🗑️  Collection '{self.collection_name}' eliminata")
