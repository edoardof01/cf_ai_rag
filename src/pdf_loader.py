"""
pdf_loader.py — Estrazione del testo grezzo dai file PDF.

Utilizza PyMuPDF (fitz) per leggere ogni pagina del PDF
e restituire il testo completo come stringa, insieme ai metadati
(nome file, numero pagina) per ogni blocco di testo.
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PageContent:
    """Rappresenta il contenuto testuale di una singola pagina PDF."""
    text: str
    page_number: int
    source_file: str


def load_pdf(file_path: str | Path) -> list[PageContent]:
    """
    Carica un file PDF e restituisce una lista di PageContent,
    uno per ogni pagina del documento.

    Args:
        file_path: Percorso al file PDF.

    Returns:
        Lista di PageContent con il testo di ogni pagina.

    Raises:
        FileNotFoundError: Se il file non esiste.
        ValueError: Se il file non è un PDF valido.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Il file non è un PDF: {path}")

    pages: list[PageContent] = []

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # Estrazione in plain text
            if text.strip():  # Ignora pagine vuote
                pages.append(PageContent(
                    text=text.strip(),
                    page_number=page_num,
                    source_file=path.name,
                ))

    return pages


def load_pdfs_from_directory(directory: str | Path) -> list[PageContent]:
    """
    Carica tutti i file PDF da una directory.

    Args:
        directory: Percorso alla directory contenente i PDF.

    Returns:
        Lista di PageContent da tutti i PDF trovati.
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Non è una directory: {dir_path}")

    all_pages: list[PageContent] = []
    pdf_files = sorted(dir_path.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  Nessun file PDF trovato in: {dir_path}")
        return all_pages

    for pdf_file in pdf_files:
        print(f"📄 Caricamento: {pdf_file.name}")
        pages = load_pdf(pdf_file)
        all_pages.extend(pages)
        print(f"   → {len(pages)} pagine estratte")

    print(f"\n✅ Totale: {len(all_pages)} pagine da {len(pdf_files)} file PDF")
    return all_pages
