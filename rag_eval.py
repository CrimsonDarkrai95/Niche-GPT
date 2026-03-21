"""
rag_eval.py — NicheGPT Evaluation Script
==========================================
Measures Precision@K, Recall@K, MRR, and per-stage latency
using your exact embedding, retrieval, and generation stack.

Usage:
  1. Place this file in the same folder as your main chatbot script
  2. Place your test PDF in the same folder (update TEST_PDF below)
  3. Run: python rag_eval.py

Requirements: same as your app (no new installs needed)
"""

import os
import time
import uuid
import json
import statistics
from pathlib import Path
from dotenv import load_dotenv

# ── Load your existing stack ──────────────────────────────────────────────────
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these two lines only
# ─────────────────────────────────────────────────────────────────────────────
TEST_PDF   = r"C:\Users\GOVIND\niche_chatgpt\data\5085589.pdf"     # PDF to evaluate against (place in same folder)
GROQ_MODEL = "llama-3.3-70b-versatile"
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv("groq_api.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── Same settings as your app ─────────────────────────────────────────────────
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
BATCH_SIZE    = 64

print("⏳ Loading embedding model (all-MiniLM-L6-v2)…")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": BATCH_SIZE},
)
print("✅ Embedding model ready\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Build ChromaDB from your test PDF (same as process_files())
# ═══════════════════════════════════════════════════════════════════════════════

def build_db(pdf_path: str) -> tuple:
    """Ingest PDF into ChromaDB — mirrors your process_files() logic exactly."""
    import pdfplumber

    path = Path(pdf_path)
    print(f"📄 Ingesting: {path.name}")

    with pdfplumber.open(path) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)

    docs = [
        Document(page_content=chunk, metadata={"source": path.name, "chunk_id": f"chunk_{i:03d}"})
        for i, chunk in enumerate(splitter.split_text(text))
    ]

    t0 = time.perf_counter()
    db = Chroma.from_documents(
        docs, embeddings,
        collection_name=f"eval_{uuid.uuid4().hex[:8]}",
    )
    ingest_time = time.perf_counter() - t0

    print(f"✅ Ingested {len(docs)} chunks in {ingest_time:.1f}s")
    print(f"   Chunk IDs: chunk_000 … chunk_{len(docs)-1:03d}\n")

    return db, docs


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Metric Functions
# ═══════════════════════════════════════════════════════════════════════════════

def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int = 3) -> float:
    hits = sum(1 for r in retrieved_ids[:k] if r in set(relevant_ids))
    return hits / k

def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int = 3) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids[:k] if r in set(relevant_ids))
    return hits / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: list, relevant_ids: list) -> float:
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Latency Measurement (mirrors your chat() function exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def measure_one_query(query: str, db, num_chunks: int, include_generation: bool = True) -> dict:
    """
    Runs one full pipeline pass and returns per-stage latency in ms.
    Mirrors your chat() function: similarity_search → Groq call.
    """
    k = 3  # fixed k=3 for eval consistency

    # ── Stage 1: Embedding (query encoding) ──────────────────────────────────
    t0 = time.perf_counter()
    # ChromaDB embeds the query internally — we time the full similarity_search
    # and subtract retrieval to isolate embedding. For accuracy we time separately:
    _ = embeddings.embed_query(query)
    embed_ms = (time.perf_counter() - t0) * 1000

    # ── Stage 2: Retrieval (ChromaDB similarity_search) ──────────────────────
    t0 = time.perf_counter()
    results = db.similarity_search(query, k=k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    retrieved_ids = [r.metadata.get("chunk_id", "") for r in results]
    context_text  = "\n\n".join(r.page_content for r in results)

    # ── Stage 3: Generation (Groq) ───────────────────────────────────────────
    gen_ms = 0.0
    answer = ""
    if include_generation and client:
        system = (
            "You are a helpful study assistant called Niche GPT. "
            "Answer using ONLY the context below. Be precise.\n\n"
            f"--- CONTEXT ---\n{context_text}\n--- END CONTEXT ---"
        )
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": query},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[Groq error: {e}]"
        gen_ms = (time.perf_counter() - t0) * 1000

    return {
        "query":         query,
        "retrieved_ids": retrieved_ids,
        "answer":        answer,
        "embed_ms":      round(embed_ms, 1),
        "retrieval_ms":  round(retrieval_ms, 1),
        "gen_ms":        round(gen_ms, 1),
        "total_ms":      round(embed_ms + retrieval_ms + gen_ms, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Eval Dataset
# Edit relevant_chunk_ids AFTER seeing which chunks were ingested above.
# Run once with SKIP_METRICS=True first to print chunk contents, then fill in.
# ═══════════════════════════════════════════════════════════════════════════════

# ── INSTRUCTIONS ──────────────────────────────────────────────────────────────
# Step 1: Run this script once → it prints all chunk IDs and first 80 chars
# Step 2: Read each chunk, pick 3-4 relevant chunk_ids per question
# Step 3: Fill in relevant_chunk_ids below and run again for final metrics
# ─────────────────────────────────────────────────────────────────────────────

EVAL_DATASET = [
    {
        "question": "What is the main topic of this document?",
        "relevant_chunk_ids": ["chunk_000", "chunk_001", "chunk_003"],
        # chunk_000: "Market Making with Fads, Informed, and Uninformed Traders"
        # chunk_001: mentions informed/uninformed traders and fads
        # chunk_003: keywords — marketmaking, signals, informed traders
    },
    {
        "question": "What are fads in the context of this paper?",
        "relevant_chunk_ids": ["chunk_006", "chunk_007"],
        # chunk_006: fad associated with market inefficiencies
        # chunk_007: fad drives mid-prices, mean-reversion
    },
    {
        "question": "What is toxic flow?",
        "relevant_chunk_ids": ["chunk_010"],
        # chunk_010: "Toxic flow is the trading activity of informed traders"
    },
    {
        "question": "What is the bid-ask pricing strategy?",
        "relevant_chunk_ids": ["chunk_009", "chunk_017"],
        # chunk_009: optimal bid-ask pricing strategies
        # chunk_017: market maker decreases price of liquidity on ask side
    },
    {
        "question": "What is the difference between full and partial information?",
        "relevant_chunk_ids": ["chunk_002", "chunk_015", "chunk_019"],
    },
    {
        "question": "Who are the uninformed traders?",
        "relevant_chunk_ids": ["chunk_001", "chunk_008", "chunk_011"],
    },
    {
        "question": "What control problem does the market maker face?",
        "relevant_chunk_ids": ["chunk_002", "chunk_009", "chunk_016"],
    },
    {
        "question": "What are the key findings about optimal strategies?",
        "relevant_chunk_ids": ["chunk_016", "chunk_017", "chunk_018"],
    },
    {
        "question": "What predictive signals are discussed?",
        "relevant_chunk_ids": ["chunk_012", "chunk_014"],
    },
    {
        "question": "What is the paper's conclusion or summary?",
        "relevant_chunk_ids": ["chunk_000", "chunk_001", "chunk_019"],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TF-IDF Baseline (no extra installs)
# ═══════════════════════════════════════════════════════════════════════════════

class TFIDFRetriever:
    def __init__(self, docs: list):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        self._cos = cosine_similarity
        self._np  = np
        self.ids   = [d.metadata["chunk_id"] for d in docs]
        self.texts = [d.page_content for d in docs]
        self.vec   = TfidfVectorizer(stop_words="english")
        self.mat   = self.vec.fit_transform(self.texts)

    def retrieve(self, query: str, k: int = 3) -> list:
        q   = self.vec.transform([query])
        sc  = self._cos(q, self.mat).flatten()
        top = self._np.argsort(sc)[::-1][:k]
        return [self.ids[i] for i in top]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("NICHEGPT — RAG EVALUATION")
    print("=" * 60)

    # ── Build DB ──────────────────────────────────────────────────────────────
    if not Path(TEST_PDF).exists():
        print(f"\n❌ '{TEST_PDF}' not found in current folder.")
        print("Place any PDF in this folder and update TEST_PDF in the script.")
        return

    db, docs = build_db(TEST_PDF)
    tfidf    = TFIDFRetriever(docs)

    # ── Print chunk map (helps fill in relevant_chunk_ids) ───────────────────
    print("── CHUNK MAP (first 80 chars each) ──────────────────────")
    for d in docs[:20]:   # print first 20 chunks
        cid     = d.metadata["chunk_id"]
        preview = d.page_content[:80].replace("\n", " ")
        print(f"  {cid}: {preview}…")
    if len(docs) > 20:
        print(f"  … and {len(docs)-20} more chunks")
    print()

    # ── Run evaluation ────────────────────────────────────────────────────────
    K = 3
    emb_precisions, emb_recalls, emb_mrrs = [], [], []
    tfi_precisions, tfi_recalls, tfi_mrrs = [], [], []
    embed_times, retrieval_times, gen_times = [], [], []

    print(f"── RUNNING {len(EVAL_DATASET)} QUERIES ──────────────────────────────")

    for i, item in enumerate(EVAL_DATASET):
        q   = item["question"]
        rel = item["relevant_chunk_ids"]

        # Embedding retrieval
        out = measure_one_query(q, db, len(docs), include_generation=(i < 3))
        embed_times.append(out["embed_ms"])
        retrieval_times.append(out["retrieval_ms"])
        if out["gen_ms"] > 0:
            gen_times.append(out["gen_ms"])

        emb_precisions.append(precision_at_k(out["retrieved_ids"], rel, K))
        emb_recalls.append(recall_at_k(out["retrieved_ids"], rel, K))
        emb_mrrs.append(mean_reciprocal_rank(out["retrieved_ids"], rel))

        # TF-IDF baseline
        tfi_ids = tfidf.retrieve(q, k=K)
        tfi_precisions.append(precision_at_k(tfi_ids, rel, K))
        tfi_recalls.append(recall_at_k(tfi_ids, rel, K))
        tfi_mrrs.append(mean_reciprocal_rank(tfi_ids, rel))

        print(f"  Q{i+1:02d}: Emb P@3={emb_precisions[-1]:.2f} | TF-IDF P@3={tfi_precisions[-1]:.2f} | {q[:50]}…")

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n── Retrieval Quality (k={K}, n={len(EVAL_DATASET)} queries) ──")
    print(f"{'Method':<25} {'Precision@3':>12} {'Recall@3':>10} {'MRR':>8}")
    print(f"{'─'*25} {'─'*12} {'─'*10} {'─'*8}")
    print(f"{'TF-IDF Baseline':<25} {statistics.mean(tfi_precisions):>12.3f} {statistics.mean(tfi_recalls):>10.3f} {statistics.mean(tfi_mrrs):>8.3f}")
    print(f"{'Embedding (MiniLM)':<25} {statistics.mean(emb_precisions):>12.3f} {statistics.mean(emb_recalls):>10.3f} {statistics.mean(emb_mrrs):>8.3f}")

    improvement = (statistics.mean(emb_precisions) - statistics.mean(tfi_precisions)) / max(statistics.mean(tfi_precisions), 0.001) * 100
    print(f"\n  Embedding vs TF-IDF: {improvement:+.1f}% Precision@3")

    print(f"\n── Latency (avg over {len(embed_times)} queries) ──")
    print(f"  Embedding :  {statistics.mean(embed_times):.1f} ms")
    print(f"  Retrieval :  {statistics.mean(retrieval_times):.1f} ms")
    if gen_times:
        print(f"  Generation:  {statistics.mean(gen_times):.0f} ms  (avg over {len(gen_times)} queries)")
        total = statistics.mean(embed_times) + statistics.mean(retrieval_times) + statistics.mean(gen_times)
        print(f"  Total     :  {total:.0f} ms")

    print(f"\n── Dataset Info ──")
    print(f"  PDF         : {TEST_PDF}")
    print(f"  Chunks      : {len(docs)}")
    print(f"  Chunk size  : {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP}")
    print(f"  Embed model : all-MiniLM-L6-v2")
    print(f"  Vector store: ChromaDB (in-memory)")
    print(f"  LLM         : {GROQ_MODEL} via Groq")

    # ── Save results to JSON ──────────────────────────────────────────────────
    results = {
        "precision_at_3": {
            "tfidf":     round(statistics.mean(tfi_precisions), 3),
            "embedding": round(statistics.mean(emb_precisions), 3),
        },
        "recall_at_3": {
            "tfidf":     round(statistics.mean(tfi_recalls), 3),
            "embedding": round(statistics.mean(emb_recalls), 3),
        },
        "mrr": {
            "tfidf":     round(statistics.mean(tfi_mrrs), 3),
            "embedding": round(statistics.mean(emb_mrrs), 3),
        },
        "latency_ms": {
            "embed_avg":    round(statistics.mean(embed_times), 1),
            "retrieval_avg": round(statistics.mean(retrieval_times), 1),
            "gen_avg":      round(statistics.mean(gen_times), 1) if gen_times else None,
        },
        "dataset": {"pdf": TEST_PDF, "chunks": len(docs), "queries": len(EVAL_DATASET)},
    }

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to eval_results.json")
    print("   Paste these numbers into your README.")
    print("=" * 60)


if __name__ == "__main__":
    main()