"""
bm25_store.py — Indice BM25 per ricerca lessicale.

BM25 (Best Matching 25) è un algoritmo di ranking lessicale che ordina
i documenti in base alla frequenza dei termini della query, ponderata
dalla rarità del termine nel corpus (IDF) e dalla lunghezza del documento.

Formula BM25:
    score(q, d) = Σ IDF(t) · [TF(t,d) · (k₁ + 1)] / [TF(t,d) + k₁ · (1 - b + b · |d|/avgdl)]

dove:
    - TF(t,d): frequenza del termine t nel documento d
    - IDF(t): inverse document frequency del termine t
    - k₁ = 1.5: parametro di saturazione della frequenza
    - b = 0.75: parametro di normalizzazione per la lunghezza del documento
    - avgdl: lunghezza media dei documenti nel corpus

Questo modulo mantiene un indice BM25 in memoria Python, con persistenza
su Redis (serializzazione pickle) per condivisione tra Pod Kubernetes.
"""

import pickle
import numpy as np

import redis
from rank_bm25 import BM25Okapi


# Stopwords italiane di base per migliorare la qualità del tokenizer.
# Rimuovendo parole funzionali ad alta frequenza, il BM25 si concentra
# sui termini semanticamente significativi.
_ITALIAN_STOPWORDS = frozenset({
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "e", "o", "ma", "che", "non", "è", "sono", "ha", "ho",
    "del", "della", "dei", "delle", "al", "alla", "ai", "alle",
    "nel", "nella", "nei", "nelle", "sul", "sulla", "sui", "sulle",
    "dal", "dalla", "dai", "dalle", "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle", "come", "cosa", "chi",
    "dove", "quando", "perché", "anche", "più", "molto", "tutto",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "of", "to", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "if", "then", "else", "when",
    "this", "that", "these", "those", "it", "its",
})

# Chiave Redis per la persistenza dell'indice BM25
_REDIS_BM25_KEY = "bm25:index"


class BM25Store:
    """
    Indice BM25 per ricerca lessicale sui chunk dei documenti.

    Gestisce il ciclo di vita dell'indice: costruzione, ricerca,
    aggiornamento incrementale e persistenza su Redis.
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.corpus: list[list[str]] = []    # Corpus tokenizzato
        self.texts: list[str] = []           # Testi originali
        self.metadata: list[dict] = []       # Metadati (source_file, page_number)

    # ------------------------------------------------------------------ #
    #                     COSTRUZIONE DELL'INDICE                         #
    # ------------------------------------------------------------------ #

    def build_index(self, texts: list[str], metadata: list[dict]) -> None:
        """
        Costruisce l'indice BM25 da zero a partire da una lista di chunk.

        Args:
            texts: Lista dei testi dei chunk.
            metadata: Lista di dizionari con 'source_file' e 'page_number'
                      per ogni chunk (stesso ordine di texts).
        """
        self.texts = texts
        self.metadata = metadata
        self.corpus = [self._tokenize(t) for t in texts]

        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            print(f"📖 [BM25] Indice costruito con {len(texts)} documenti")
        else:
            self.bm25 = None
            print("⚠️ [BM25] Nessun documento da indicizzare")

    def add_documents(self, texts: list[str], metadata: list[dict]) -> None:
        """
        Aggiunge nuovi documenti all'indice esistente e lo ricostruisce.

        Il BM25 deve essere ricostruito interamente perché le statistiche
        corpus-level (IDF, avgdl) cambiano con ogni nuovo documento.

        Args:
            texts: Nuovi testi da aggiungere.
            metadata: Metadati corrispondenti.
        """
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        new_tokenized = [self._tokenize(t) for t in texts]
        self.corpus.extend(new_tokenized)

        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            print(f"📖 [BM25] Indice aggiornato: {len(self.texts)} documenti totali (+{len(texts)} nuovi)")

    # ------------------------------------------------------------------ #
    #                           RICERCA                                   #
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Cerca i chunk più rilevanti per la query usando BM25.

        Args:
            query: La query testuale.
            top_k: Numero di risultati da restituire.

        Returns:
            Lista di dizionari con 'text', 'source_file', 'page_number', 'score'.
            Ordinata per score decrescente. Esclusi risultati con score 0.
        """
        if not self.bm25 or not self.corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Prendi solo i top_k con score > 0 (nessun match = score 0)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": self.texts[idx],
                    "source_file": self.metadata[idx]["source_file"],
                    "page_number": self.metadata[idx]["page_number"],
                    "score": float(scores[idx]),
                })
        return results

    # ------------------------------------------------------------------ #
    #                      TOKENIZZAZIONE                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenizza un testo per BM25.

        Strategia semplice ma efficace:
          1. Converte in minuscolo
          2. Divide per spazi e punteggiatura
          3. Rimuove token troppo corti (< 2 chars)
          4. Rimuove stopwords italiane e inglesi

        Args:
            text: Il testo da tokenizzare.

        Returns:
            Lista di token filtrati.
        """
        # Sostituisci punteggiatura con spazi, poi split
        cleaned = ""
        for char in text.lower():
            if char.isalnum() or char == " ":
                cleaned += char
            else:
                cleaned += " "

        tokens = cleaned.split()
        return [t for t in tokens if len(t) > 1 and t not in _ITALIAN_STOPWORDS]

    # ------------------------------------------------------------------ #
    #                   PERSISTENZA SU REDIS                              #
    # ------------------------------------------------------------------ #

    def save_to_redis(self, redis_client: redis.Redis) -> None:
        """
        Serializza l'indice BM25 su Redis per persistenza multi-pod.

        Salva solo i dati grezzi (testi, metadati, corpus tokenizzato).
        L'oggetto BM25Okapi viene ricostruito al caricamento perché
        non è serializzabile direttamente.
        """
        data = {
            "texts": self.texts,
            "metadata": self.metadata,
            "corpus": self.corpus,
        }
        redis_client.set(_REDIS_BM25_KEY, pickle.dumps(data))
        print(f"💾 [BM25] Indice salvato su Redis ({len(self.texts)} documenti)")

    def load_from_redis(self, redis_client: redis.Redis) -> bool:
        """
        Carica l'indice BM25 da Redis.

        Returns:
            True se l'indice è stato caricato con successo, False altrimenti.
        """
        raw = redis_client.get(_REDIS_BM25_KEY)
        if not raw:
            return False

        data = pickle.loads(raw)
        self.texts = data["texts"]
        self.metadata = data["metadata"]
        self.corpus = data["corpus"]

        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            print(f"📖 [BM25] Indice caricato da Redis ({len(self.texts)} documenti)")
            return True
        return False

    # ------------------------------------------------------------------ #
    #                           UTILITY                                   #
    # ------------------------------------------------------------------ #

    def is_empty(self) -> bool:
        """Restituisce True se l'indice non contiene documenti."""
        return len(self.texts) == 0

    def count(self) -> int:
        """Restituisce il numero di documenti nell'indice."""
        return len(self.texts)
