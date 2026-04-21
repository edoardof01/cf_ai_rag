"""
reranker.py — Modulo per il Re-Ranking dei documenti tramite Cross-Encoder.

Il Re-Ranking (o "Stage 2 Retrieval") prende una lista di documenti
candidati (recuperati tramite metodi veloci come Vector Search e BM25)
e li ri-ordina usando un modello neurale più accurato e computazionalmente
più pesante.

Un Cross-Encoder valuta simultaneamente la query e il documento,
calcolando uno score di rilevanza semantica profonda (da 0 a 1),
cogliendo interazioni logiche che i semplici Bi-Encoder (embedding) perdono.
"""

from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderReranker:
    """
    Reranker basato su Cross-Encoder di HuggingFace.
    """

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        """
        Inizializza il modello Cross-Encoder.
        
        Args:
            model_name: Nome del modello su HuggingFace. Il default è un
                        modello multilingua (ottimo per italiano) basato
                        su MiniLM, veloce e leggero.
        """
        print(f"🔄 Caricamento modello di Re-Ranking (Cross-Encoder): {model_name}...")
        self.model = CrossEncoder(model_name)
        print("✅ Modello di Re-Ranking caricato.")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        Ri-ordina una lista di documenti candidati in base alla loro
        reale rilevanza semantica rispetto alla query.

        Args:
            query: La domanda dell'utente.
            candidates: Lista di dizionari (i risultati di BM25/Vector store).
                        Devono contenere la chiave 'text'.
            top_k: Quanti documenti restituire dopo il ri-ordinamento.

        Returns:
            Lista ristretta (top_k) e ri-ordinata di candidati.
        """
        if not candidates:
            return []

        # Costruiamo le coppie (Query, Documento) richieste dal Cross-Encoder
        cross_inp = [[query, doc["text"]] for doc in candidates]

        # Calcoliamo gli score
        # I modelli Cross-Encoder restituiscono logit grezzi. Non serve
        # convertirli con sigmoide perché a noi interessa solo l'ordinamento relativo.
        scores = self.model.predict(cross_inp)

        # Aggiungiamo lo score di reranking ai documenti
        for idx, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[idx])

        # Ordiniamo in ordine decrescente rispetto al nuovo score
        reranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return reranked_candidates[:top_k]
