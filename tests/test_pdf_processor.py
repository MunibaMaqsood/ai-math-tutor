from langchain_core.documents import Document
from src.pdf_processor import clean_text, split_documents


def test_clean_text_collapses_blank_lines():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_split_keeps_metadata():
    doc = Document(page_content="word " * 1000, metadata={"source": "x.pdf", "page": 3})
    chunks = split_documents([doc])
    assert len(chunks) > 1
    assert all(c.metadata["page"] == 3 for c in chunks)
