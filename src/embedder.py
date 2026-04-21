"""
embedder.py — Generazione di embedding vettoriali dal testo.

Un embedding è una rappresentazione numerica (vettore) di un testo,
dove testi con significato simile producono vettori vicini nello
spazio multidimensionale. Questo permette di cercare documenti
per "vicinanza semantica" anziché per corrispondenza esatta di parole.

Fase 1: Utilizza un modello pre-trained di sentence-transformers.
Fase 5 (futura): Sostituiremo con un modello custom.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    Wrapper per la generazione di embedding vettoriali.

    Incapsula il modello di embedding dietro un'interfaccia semplice,
    così che in futuro possiamo sostituire il modello pre-trained
    con uno custom senza cambiare il resto del codice.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inizializza il modello di embedding.

        Args:
            model_name: Nome del modello sentence-transformers da usare.
                        'all-MiniLM-L6-v2' produce vettori a 384 dimensioni,
                        è leggero (~80MB) e ha buone performance multilingua.
        """
        print(f"🔄 Caricamento modello di embedding: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Modello caricato. Dimensione vettori: {self.dimension}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Genera l'embedding per un singolo testo.

        Args:
            text: Il testo da convertire in vettore.

        Returns:
            Vettore numpy di dimensione (self.dimension,).
        """
        return self.model.encode(text, normalize_embeddings=True)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Genera gli embedding per una lista di testi (batch processing).

        Il batch processing è molto più efficiente che chiamare
        embed_text() in un loop, perché sfrutta il parallelismo della GPU/CPU.

        Args:
            texts: Lista di testi da convertire.
            batch_size: Numero di testi da processare simultaneamente.

        Returns:
            Matrice numpy di dimensione (len(texts), self.dimension).
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def get_dimension(self) -> int:
        """Restituisce la dimensione dei vettori di embedding."""
        return self.dimension
