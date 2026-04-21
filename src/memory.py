"""
memory.py — Memoria conversazionale multi-strategia con persistenza.

Implementa tre forme di memoria con storage distribuito:
  1. Key-Value (KV):  Redis List  — finestra scorrevole degli ultimi W turni.
  2. Summary (S):     Redis String — riassunto cumulativo, aggiornato ogni N turni.
  3. Vector (VM):     Milvus Collection dedicata — embedding dei turni per ricerca semantica.
                      Ogni turno viene riscritto dall'LLM come standalone query
                      prima della vettorizzazione (Query Rewriting).

Storage:
  - Redis keys per sessione:
      session:{id}:kv             → Lista JSON dei turni recenti
      session:{id}:summary        → Stringa del riassunto cumulativo
      session:{id}:turns          → Contatore intero dei turni totali
      session:{id}:summary_buffer → Buffer dei turni non ancora riassunti

  - Milvus collection: rag_conversation_memory
      session_id (VARCHAR) | turn_text (VARCHAR) | embedding (FLOAT_VECTOR)

Fornisce inoltre:
  - φ_hybrid(M_{t-1}): query extension combinando Summary + Vector retrieval
  - estimate_tokens(): euristica per il conteggio token (~4 chars/token)
"""

import json
import numpy as np
import redis

from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility,
)

from src.llm_client import OllamaClient


# --------------------------------------------------------------------- #
#                         COSTANTI E TEMPLATE                            #
# --------------------------------------------------------------------- #

# Template usato dall'LLM per aggiornare il riassunto cumulativo.
# In questa versione, il buffer accumula più scambi tra un aggiornamento e l'altro.
SUMMARY_UPDATE_TEMPLATE = """Aggiorna il seguente riassunto della conversazione includendo i nuovi scambi.
Mantieni il riassunto conciso (max 200 parole), focalizzandoti sugli argomenti chiave discussi.

RIASSUNTO ATTUALE:
{current_summary}

NUOVI SCAMBI:
{exchanges}

RIASSUNTO AGGIORNATO:"""

# Template per il Query Rewriting dei turni prima della vettorizzazione.
# Invece di embeddare il testo grezzo "Domanda: ... Risposta: ...",
# l'LLM produce una standalone query decontestualizzata che cattura
# l'essenza semantica del turno in modo compatto e ricercabile.
TURN_REWRITE_TEMPLATE = """Dato il seguente scambio di una conversazione, riscrivi in una singola frase \
chiara e autonoma il concetto chiave discusso. La frase deve essere comprensibile \
senza contesto esterno e deve catturare sia la domanda che la risposta.
Non aggiungere commenti, scrivi solo la frase riscritta.

Domanda: {question}
Risposta: {answer}

Frase riscritta:"""

# Nome della collection Milvus dedicata alla memoria conversazionale
MEMORY_COLLECTION_NAME = "rag_conversation_memory"


# --------------------------------------------------------------------- #
#                         TOKEN ESTIMATION                               #
# --------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    """
    Stima approssimativa del numero di token in un testo.

    Euristica: ~4 caratteri per token (standard per tokenizer BPE
    come quello di Llama3, che produce in media 3.5-4.5 chars/token
    per testo in lingue europee).

    Args:
        text: Il testo di cui stimare i token.

    Returns:
        Numero stimato di token.
    """
    if not text:
        return 0
    return len(text) // 4


# --------------------------------------------------------------------- #
#                     CONVERSATION MEMORY (PERSISTENTE)                   #
# --------------------------------------------------------------------- #

class ConversationMemory:
    """
    Memoria conversazionale persistente su Redis + Milvus.

    Ogni istanza è legata a un session_id e opera su dati esterni
    (Redis/Milvus), non su stato interno Python. Questo permette a
    più Pod Kubernetes di condividere la stessa memoria di sessione.
    """

    def __init__(
        self,
        session_id: str,
        redis_client: redis.Redis,
        llm_client: OllamaClient,
        embedder,
        embedding_dim: int = 384,
        kv_window: int = 5,
        summary_update_interval: int = 5,
    ):
        """
        Args:
            session_id: Identificativo univoco della sessione.
            redis_client: Connessione Redis già inizializzata.
            llm_client: Client Ollama per aggiornare il summary via LLM.
            embedder: Istanza di Embedder per generare gli embedding dei turni.
            embedding_dim: Dimensione dei vettori di embedding.
            kv_window: Numero massimo di turni nella KV memory (FIFO).
            summary_update_interval: Ogni quanti turni aggiornare il summary.
        """
        self.session_id = session_id
        self.redis = redis_client
        self.llm_client = llm_client
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.kv_window = kv_window
        self.summary_update_interval = summary_update_interval

        # Redis keys per questa sessione
        self._kv_key = f"session:{session_id}:kv"
        self._summary_key = f"session:{session_id}:summary"
        self._turns_key = f"session:{session_id}:turns"
        self._summary_buffer_key = f"session:{session_id}:summary_buffer"

        # Inizializza la collection Milvus per la Vector Memory (se non esiste)
        self._init_milvus_collection()

    # ------------------------------------------------------------------ #
    #               INIZIALIZZAZIONE MILVUS (Vector Memory)               #
    # ------------------------------------------------------------------ #

    def _init_milvus_collection(self) -> None:
        """Crea la collection Milvus per la memoria conversazionale se non esiste."""
        if utility.has_collection(MEMORY_COLLECTION_NAME):
            self._collection = Collection(MEMORY_COLLECTION_NAME)
            self._collection.load()
            return

        print(f"🆕 Creazione collection Milvus '{MEMORY_COLLECTION_NAME}' per la memoria...")
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("session_id", DataType.VARCHAR, max_length=256),
            FieldSchema("turn_text", DataType.VARCHAR, max_length=65535),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]
        schema = CollectionSchema(fields, description="Conversational memory vectors")
        self._collection = Collection(MEMORY_COLLECTION_NAME, schema)

        self._collection.create_index("embedding", {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        })
        print(f"✅ Collection '{MEMORY_COLLECTION_NAME}' creata con indice IVF_FLAT")
        self._collection.load()

    # ------------------------------------------------------------------ #
    #                       AGGIORNAMENTO MEMORIA                         #
    # ------------------------------------------------------------------ #

    def add_turn(self, question: str, answer: str) -> None:
        """
        Aggiorna tutte e tre le memorie dopo un turno di conversazione.

        Operazione: M_t = f(M_{t-1}, u_t, a_t)
          1. KV: push su Redis List + trim a W elementi (FIFO).
          2. Summary: bufferizza lo scambio; aggiorna il riassunto ogni N turni.
          3. Vector: Query Rewriting via LLM → embedding → salva in Milvus.
        """
        # 1. KV Memory — Redis List (FIFO con finestra W)
        turn_json = json.dumps({"q": question, "a": answer})
        self.redis.rpush(self._kv_key, turn_json)
        self.redis.ltrim(self._kv_key, -self.kv_window, -1)

        # 2. Incrementa il contatore globale dei turni
        turns = self.redis.incr(self._turns_key)

        # 3. Summary — Bufferizza lo scambio
        buffer_entry = f"Utente: {question}\nAssistente: {answer}"
        self.redis.rpush(self._summary_buffer_key, buffer_entry)

        # Aggiorna il summary solo ogni N turni (evita degradazione)
        if turns % self.summary_update_interval == 0:
            self._update_summary()

        # 4. Vector Memory — Query Rewriting + Embedding
        # Invece di embeddare il testo grezzo "Domanda: ... Risposta: ...",
        # facciamo riscrivere il turno dall'LLM in una standalone query
        # decontestualizzata. Questo migliora l'allineamento semantico
        # durante il retrieval per cosine similarity.
        rewritten = self._rewrite_turn(question, answer)
        turn_embedding = self.embedder.embed_text(rewritten)
        self._store_vector(rewritten, turn_embedding)

    def _rewrite_turn(self, question: str, answer: str) -> str:
        """
        Riscrive un turno Q&A in una standalone query decontestualizzata
        tramite l'LLM, per ottenere un embedding più pulito e ricercabile.

        Fallback: se l'LLM fallisce, usa il formato grezzo originale.

        Args:
            question: La domanda dell'utente.
            answer: La risposta dell'assistente.

        Returns:
            Stringa standalone riscritta (o fallback grezzo).
        """
        fallback = f"Domanda: {question}\nRisposta: {answer}"

        try:
            prompt = TURN_REWRITE_TEMPLATE.format(
                question=question,
                answer=answer,
            )
            rewritten = self.llm_client.generate(prompt, stream=False)
            # Sanity check: se l'LLM restituisce qualcosa di vuoto o troppo lungo, fallback
            if rewritten and len(rewritten.strip()) > 5 and len(rewritten) < 500:
                return rewritten.strip()
            return fallback
        except Exception as e:
            print(f"⚠️ [Memory] Query rewriting fallito, uso fallback: {e}")
            return fallback

    def _update_summary(self) -> None:
        """
        Aggiorna il riassunto cumulativo usando tutti gli scambi bufferizzati.
        Viene chiamato ogni summary_update_interval turni, non ad ogni turno,
        per mitigare il drift semantico da riscritture eccessive.
        """
        # Recupera il buffer degli scambi non ancora riassunti
        raw_buffer = self.redis.lrange(self._summary_buffer_key, 0, -1)
        if not raw_buffer:
            return

        buffered_text = "\n\n".join([entry.decode("utf-8") for entry in raw_buffer])
        current_summary = self.get_summary()

        prompt = SUMMARY_UPDATE_TEMPLATE.format(
            current_summary=current_summary if current_summary else "(Conversazione appena iniziata)",
            exchanges=buffered_text,
        )

        try:
            new_summary = self.llm_client.generate(prompt, stream=False)
            self.redis.set(self._summary_key, new_summary)
            # Svuota il buffer dopo l'aggiornamento
            self.redis.delete(self._summary_buffer_key)
        except Exception as e:
            print(f"⚠️ [Memory] Errore aggiornamento summary: {e}")

    def _store_vector(self, turn_text: str, turn_embedding: np.ndarray) -> None:
        """Salva un turno come vettore nella collection Milvus."""
        self._collection.insert([
            [self.session_id],          # session_id
            [turn_text],                # turn_text
            [turn_embedding.tolist()],  # embedding
        ])
        self._collection.flush()

    # ------------------------------------------------------------------ #
    #                          LETTURA MEMORIA                            #
    # ------------------------------------------------------------------ #

    def get_kv_history(self) -> str:
        """
        Legge da Redis gli ultimi W turni e li formatta come stringa
        per il prompt dell'LLM.
        """
        raw = self.redis.lrange(self._kv_key, 0, -1)
        if not raw:
            return ""

        parts = []
        for entry in raw:
            turn = json.loads(entry.decode("utf-8"))
            parts.append(f"Utente: {turn['q']}\nAssistente: {turn['a']}")
        return "\n\n".join(parts)

    def get_summary(self) -> str:
        """Legge da Redis il summary cumulativo corrente."""
        summary = self.redis.get(self._summary_key)
        return summary.decode("utf-8") if summary else ""

    def search_similar_turns(self, query_embedding: np.ndarray, top_k: int = 1) -> list[str]:
        """
        Cerca nella Vector Memory (Milvus) i turni semanticamente più simili.
        Filtra per session_id per isolare le sessioni.

        Args:
            query_embedding: Vettore della query corrente.
            top_k: Numero di turni da restituire.

        Returns:
            Lista dei testi dei turni più simili.
        """
        if self._collection.num_entities == 0:
            return []

        results = self._collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=top_k,
            # Filtra solo i turni di questa sessione
            expr=f'session_id == "{self.session_id}"',
            output_fields=["turn_text"],
        )

        texts = []
        for hits in results:
            for hit in hits:
                texts.append(hit.entity.get("turn_text"))
        return texts

    # ------------------------------------------------------------------ #
    #                    FUNZIONE φ (QUERY EXTENSION)                     #
    # ------------------------------------------------------------------ #

    def compute_phi(self, question: str, query_embedding: np.ndarray) -> str:
        """
        Funzione φ_hybrid(M_{t-1}): combina Summary + Vector retrieval.

        Formula:
            q'_t = q_t ⊕ φ(M_{t-1})
            φ(M_{t-1}) = "[Riassunto: S_{t-1}] [Turno correlato: VM.search(q_t, 1)]"

        Args:
            question: La domanda originale dell'utente (q_t).
            query_embedding: L'embedding della domanda.

        Returns:
            Stringa da concatenare alla query per la ricerca arricchita.
            Stringa vuota se la memoria è ancora vuota.
        """
        parts = []

        # Componente Summary (φ_text)
        summary = self.get_summary()
        if summary:
            parts.append(f"[Riassunto conversazione: {summary}]")

        # Componente Vector retrieval (φ_vec)
        similar_turns = self.search_similar_turns(query_embedding, top_k=1)
        if similar_turns:
            parts.append(f"[Turno correlato: {similar_turns[0]}]")

        if not parts:
            return ""

        return " " + " ".join(parts)

    # ------------------------------------------------------------------ #
    #                           UTILITY                                   #
    # ------------------------------------------------------------------ #

    def is_empty(self) -> bool:
        """Restituisce True se la memoria non contiene turni."""
        turns = self.redis.get(self._turns_key)
        return turns is None or int(turns) == 0

    def clear(self) -> None:
        """Azzera completamente la memoria della sessione (Redis + Milvus)."""
        # Pulisci Redis
        self.redis.delete(
            self._kv_key,
            self._summary_key,
            self._turns_key,
            self._summary_buffer_key,
        )
        # Pulisci i vettori di questa sessione da Milvus
        self._collection.delete(expr=f'session_id == "{self.session_id}"')
        self._collection.flush()

    def turn_count(self) -> int:
        """Restituisce il numero totale di turni memorizzati."""
        turns = self.redis.get(self._turns_key)
        return int(turns) if turns else 0
