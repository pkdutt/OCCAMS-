# RAG System — Scraper + Indexer + Chat App

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using a **web scraper**, **indexer**, and a **Streamlit-based chat app**.

It allows you to:
1. Scrape a website and extract all text
2. Index the extracted content using ChromaDB + OpenAI embeddings
3. Chat with the data using a RAG workflow
4. Use offline fallback mode when internet is unavailable

---

## 1. Scraper
The scraper uses **Selenium** to crawl a website, extract visible text from each page, and recursively follow internal links.

### What it does
- Loads the starting URL using Selenium
- Extracts visible text using BeautifulSoup
- Recursively visits links within the same domain
- Skips links containing `#` to avoid duplicates
- Saves all scraped text into `scraped_content.json` in the format:

```json
{
  "https://example.com": "page text...",
  "https://example.com/about": "page text..."
}
```

### How to run
```bash
python scraper.py
```

---

## 2. Indexer — `index_documents.py`
The indexer processes scraped data and prepares it for retrieval.

### What it does
- Loads `scraped_content.json`
- Converts content into LangChain `Document` objects
- Splits text into overlapping chunks for better searchability
- Generates embeddings using **OpenAI embeddings**
- Stores everything in **ChromaDB** for fast retrieval
- Saves chunked docs into `chunked_docs.json` for offline fallback

### How to run
```bash
python index_documents.py
```

Run this once before using the chat app.

---

## 3. RAG Chat App — `rag_app.py`
The chat interface allows users to have natural conversations grounded in the indexed content.

### Features
- Chat-based onboarding (Name → Email → Phone)
- Retrieves relevant chunks from ChromaDB
- Uses **OpenAI** chat models for final responses
- Offline fallback mode with keyword search
- Clean Streamlit chat UI with history
- Shows retrieved context for transparency

### How to run
```bash
streamlit run rag_app.py
```

### Environment Variables
```bash
export OPENAI_API_KEY="your-key"
```

Force offline mode:
```bash
export OFFLINE_MODE=1
```
