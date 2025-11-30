"""
index_documents.py

Read a JSON file {url: content}, chunk the text, create embeddings using
OpenAI embeddings, store everything in ChromaDB, and also dump chunked docs
to a JSON file for offline fallback.
"""

import json
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()  # load OPENAI_API_KEY from .env

# ---------- CONFIG ----------
SCRAPED_JSON = "scraped_content.json"   # <- path to your scraped JSON
CHROMA_DIR = "chroma_db"               # <- where Chroma will persist its data
CHUNKED_DOCS_JSON = "chunked_docs.json"  # used by offline fallback

# You can override default model via env var:
#   export OPENAI_EMBED_MODEL="text-embedding-3-small"
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def load_json_documents(path: str) -> List[Document]:
    """Load {url: text} from JSON and turn them into LangChain Documents."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Document] = []
    for url, content in data.items():
        if not content:
            continue
        docs.append(
            Document(
                page_content=content,
                metadata={"source": url},
            )
        )
    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split long docs into smaller overlapping chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def build_vectorstore(docs: List[Document]) -> None:
    """Create Chroma vectorstore with OpenAI embeddings and persist it."""
    print(f"Using embedding model: {EMBED_MODEL}")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    print(f"Creating ChromaDB at: {CHROMA_DIR}")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vectordb.persist()
    print("ChromaDB persisted successfully.")


def dump_chunked_docs(docs: List[Document]) -> None:
    """Dump chunked docs to JSON for offline lexical fallback."""
    serializable = []
    for d in docs:
        serializable.append(
            {
                "page_content": d.page_content,
                "metadata": d.metadata,
            }
        )

    with open(CHUNKED_DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(serializable)} chunked docs to {CHUNKED_DOCS_JSON} for offline fallback.")


def main():
    print(f"Loading documents from {SCRAPED_JSON} ...")
    docs = load_json_documents(SCRAPED_JSON)
    print(f"Loaded {len(docs)} base documents.")

    print("Chunking documents ...")
    chunked_docs = chunk_documents(docs)
    print(f"Generated {len(chunked_docs)} chunks.")

    print("Building vectorstore ...")
    build_vectorstore(chunked_docs)

    print("Dumping chunked docs for offline mode ...")
    dump_chunked_docs(chunked_docs)

    print("✅ Indexing complete.")


if __name__ == "__main__":
    # Needs OPENAI_API_KEY env var for embeddings at index time
    # export OPENAI_API_KEY="sk-..."
    main()
