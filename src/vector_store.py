"""Stage 1: embeddings + ChromaDB storage and retrieval."""
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config


def get_vector_store() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


def add_chunks(chunks: list[Document], batch_size: int = 50) -> None:
    """Embed and store chunks in batches (avoids API rate limits)."""
    store = get_vector_store()
    for i in range(0, len(chunks), batch_size):
        store.add_documents(chunks[i:i + batch_size])
        time.sleep(0.5)


def list_sources() -> list[str]:
    data = get_vector_store().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"] if m and "source" in m})


def delete_source(filename: str) -> None:
    get_vector_store().delete(where={"source": filename})


def retrieve(query: str, k: int = config.TOP_K) -> list[Document]:
    return get_vector_store().similarity_search(query, k=k)
