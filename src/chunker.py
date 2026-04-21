"""
chunker.py — Suddivisione del testo in chunk di dimensione fissa.

Implementa una strategia di "fixed-size chunking" con overlap:
il testo viene spezzato in blocchi di `chunk_size` caratteri,
con una sovrapposizione di `chunk_overlap` caratteri tra un
blocco e il successivo, per evitare di tagliare concetti a metà.
"""

from dataclasses import dataclass


@dataclass
class TextChunk:
    """Rappresenta un singolo chunk di testo con i suoi metadati."""
    text: str
    chunk_index: int
    source_file: str
    page_number: int  # Pagina di origine (del primo carattere del chunk)


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source_file: str = "",
    page_number: int = 0,
) -> list[TextChunk]:
    """
    Divide un testo in chunk di dimensione fissa con overlap.

    L'algoritmo scorre il testo con un passo di (chunk_size - chunk_overlap),
    creando finestre sovrapposte. Questo garantisce che le informazioni
    al confine tra due chunk non vengano perse.

    Esempio con chunk_size=10, overlap=3:
        Testo:   "ABCDEFGHIJKLMNOP"
        Chunk 0: "ABCDEFGHIJ"       (pos 0-9)
        Chunk 1: "HIJKLMNOP"        (pos 7-15, overlap di "HIJ")

    Args:
        text: Il testo da dividere in chunk.
        chunk_size: Dimensione massima di ogni chunk in caratteri.
        chunk_overlap: Numero di caratteri di sovrapposizione tra chunk adiacenti.
        source_file: Nome del file sorgente (per i metadati).
        page_number: Numero di pagina di origine.

    Returns:
        Lista di TextChunk.
    """
    if not text or not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) deve essere minore di "
            f"chunk_size ({chunk_size})"
        )

    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap  # Passo di avanzamento
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_content = text[start:end].strip()

        if chunk_text_content:  # Ignora chunk vuoti
            chunks.append(TextChunk(
                text=chunk_text_content,
                chunk_index=chunk_index,
                source_file=source_file,
                page_number=page_number,
            ))
            chunk_index += 1

        start += step

    return chunks


def chunk_pages(
    pages: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """
    Applica il chunking a una lista di PageContent (output di pdf_loader).

    Ogni pagina viene trattata indipendentemente: i chunk di una pagina
    non si sovrappongono con quelli della pagina successiva.

    Args:
        pages: Lista di PageContent (da pdf_loader.load_pdf).
        chunk_size: Dimensione di ogni chunk.
        chunk_overlap: Sovrapposizione tra chunk.

    Returns:
        Lista di TextChunk da tutte le pagine.
    """
    all_chunks: list[TextChunk] = []

    for page in pages:
        page_chunks = chunk_text(
            text=page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_file=page.source_file,
            page_number=page.page_number,
        )
        all_chunks.extend(page_chunks)

    return all_chunks
