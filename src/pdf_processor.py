"""Stage 1: read a PDF page by page and split it into chunks."""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config


def clean_text(text: str) -> str:
    """Normalise whitespace but keep line breaks (equations often span lines)."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[Document]:
    """Return one Document per non-empty page, tagged with file name and page number."""
    reader = PdfReader(str(pdf_path))
    docs = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": Path(pdf_path).name, "page": number},
            ))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks. Overlap keeps a formula and its
    explanation together when a chunk boundary falls between them."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def process_pdf(pdf_path: Path) -> list[Document]:
    """Full pipeline for one file: extract -> split."""
    return split_documents(extract_pages(pdf_path))
