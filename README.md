# 🤖 Niche GPT — RAG Study Assistant

> *Upload your documents. Ask anything. Get precise, source-backed answers.*

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/Emet018/Niche-GPT)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-6.8.0-orange)](https://gradio.app)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-green)](https://groq.com)

---

## What is Niche GPT?

Niche GPT is a **Retrieval-Augmented Generation (RAG)** chatbot that lets users upload their own PDF or TXT documents and ask questions about them — getting precise, source-cited answers grounded strictly in their uploaded content.

Think of it like a **NotebookLM alternative** — but engineered for precision, accuracy, and study use cases. The chatbot does not answer from general knowledge. Every answer is retrieved from the user's own documents, cited, and grounded.

---

## How It Works — Full Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                   │
│  User Uploads PDF/TXT                                            │
│         │                                                         │
│         ▼                                                         │
│  pdfplumber extracts raw text                                    │
│         │                                                         │
│         ▼                                                         │
│  RecursiveCharacterTextSplitter                                  │
│  chunk_size=400 · chunk_overlap=80                               │
│         │                                                         │
│         ▼                                                         │
│  HuggingFace MiniLM-L6-v2 Embeddings                            │
│  batch_size=64 for speed                                         │
│         │                                                         │
│         ▼                                                         │
│  ChromaDB In-Memory Vector Store (per session)                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL PIPELINE                        │
│                                                                   │
│  User asks a question                                            │
│         │                                                         │
│         ▼                                                         │
│  Dynamic K selection based on chunk count                        │
│  < 100 chunks  → K=4  (short files)                             │
│  < 500 chunks  → K=6  (medium files)                            │
│  < 1000 chunks → K=8  (long files ~50 pages)                   │
│  1000+ chunks  → K=10 (large multi-file sessions)               │
│         │                                                         │
│         ▼                                                         │
│  Semantic similarity search → Top K chunks retrieved             │
│         │                                                         │
│         ▼                                                         │
│  Chunks injected into LLaMA 3.3 70B system prompt               │
│         │                                                         │
│         ▼                                                         │
│  Answer generated · Sources cited · Returned to user            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- 📁 **Multi-file upload** — PDF and TXT support, multiple files per session
- 🔒 **Session-isolated** — each user's documents are private, in-memory only, never stored
- ⏭️ **Duplicate detection** — re-uploading the same file skips it instantly, no RAM waste
- 📊 **Dynamic K retrieval** — automatically scales retrieved chunks based on document size
- 🎯 **Source attribution** — every answer cites exactly which file it came from
- ⚡ **Groq-powered** — sub-5 second responses using Groq's LPU inference hardware
- 🧠 **LLaMA 3.3 70B** — large reasoning model for intent-aware, depth-calibrated answers

---

## Why LLaMA 3.3 70B — The Model Selection Journey

Multiple models were evaluated before settling on `llama-3.3-70b-versatile`:

| Model | Size | Speed | Reasoning | Verdict |
|---|---|---|---|---|
| `llama-3.1-8b-instant` | 8B | ~1-2s | Basic | Fast but shallow — no intent inference, fixed answer depth |
| `llama-3.1-70b-versatile` | 70B | ~3-5s | Strong | Decommissioned by Groq |
| `llama-3.3-70b-versatile` ✅ | 70B | ~3-5s | Strongest | Current model — best reasoning, newer architecture |

**Why not 8B?** Small models answer *what* to say but cannot reason about *how much* to say. A student asking "briefly explain X" gets the same length response as "explain X in detail." The 70B model naturally calibrates answer depth to question intent.

**Why not a larger model?** 70B on Groq's LPU hardware responds in 3-5 seconds — acceptable for a study tool. Models above 70B have no meaningful accuracy gain for document Q&A tasks and significantly slower response times.

---

## Precision & Accuracy Engineering

### Chunk Configuration — The Core Tradeoff

Chunk size is the single most impactful RAG parameter:

| Chunk Size | Precision | RAM Usage | Best For |
|---|---|---|---|
| 200 | Very High | High | Dense technical text |
| **400** ✅ | **High** | **Medium** | **Academic documents** |
| 600 | Medium | Low | General text |
| 1000 | Low | Very Low | Summaries only |

**Chosen:** `chunk_size=400` with `chunk_overlap=80`

The 80-token overlap ensures that sentences split across chunk boundaries are never lost — a critical fix for academic text where a single sentence can span 60-80 words.

### Dynamic K — Why Fixed K Fails

A fixed `K=5` works for a 5-page document. For a 50-page document with 500 chunks, `K=5` retrieves only 1% of available content — statistically likely to miss the answer entirely.

Dynamic K solves this by scaling retrieval to document volume:

```
85 chunks  (2 short files)  → K=4   precise, low noise
350 chunks (1 medium PDF)   → K=6   broader coverage
800 chunks (50-page doc)    → K=8   wide net
1200 chunks (many files)    → K=10  maximum coverage
```

### Embedding Model

`all-MiniLM-L6-v2` was chosen over alternatives for the following reasons:

- **Speed:** 5x faster than `all-mpnet-base-v2` with only ~2% accuracy loss
- **Size:** 90MB — fits comfortably in HF Spaces free RAM
- **Batch processing:** `batch_size=64` on 16GB RAM gives ~40-50% speed improvement over default sequential embedding
- **Quality:** Trained specifically for semantic similarity tasks — ideal for Q&A retrieval

---

## Session Limits & Safety

| Limit | Value | Reason |
|---|---|---|
| Max files | 25 | Prevents RAM exhaustion on 16GB HF Spaces |
| Max session size | 200MB | Safe ceiling with embedding model loaded |
| Max per-file size | 20MB | Prevents single large file from consuming all RAM |
| Max output tokens | 1500 | Enough for multi-question answers without waste |
| History window | Last 6 turns | Keeps context relevant without bloating prompt |

---

## Installation & Local Setup

```bash
git clone https://github.com/CrimsonDarkrai95/Niche-GPT
cd Niche-GPT
pip install -r requirements.txt
```

Create a `groq_api.env` file:
```
GROQ_API_KEY=your_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

```bash
python app.py
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | HuggingFace MiniLM-L6-v2 |
| Vector Store | ChromaDB (in-memory) |
| PDF Parsing | pdfplumber |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Frontend | Gradio 6.8.0 |
| Hosting | HuggingFace Spaces (2 vCPU, 16GB RAM) |

---

## Architecture Decisions

**Why ChromaDB in-memory vs persistent?**
User privacy. Documents exist only for the session duration. No data is written to disk, no cross-session contamination is possible.

**Why Gradio over a custom frontend?**
Gradio runs natively on HuggingFace Spaces with zero infrastructure overhead. A React/Three.js frontend would require a separate host (Vercel) and an API layer — unnecessary complexity for the current scope.

**Why pdfplumber over PyPDF2?**
pdfplumber handles complex PDF layouts, tables, and multi-column academic papers significantly better than PyPDF2, which frequently garbles text extraction from research PDFs.

---

*Built with the goal of making dense academic content accessible — one question at a time.*