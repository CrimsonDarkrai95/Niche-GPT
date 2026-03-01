"""
Niche GPT — RAG Chatbot
Uses Groq (free Llama 3) + ChromaDB + Gradio 6.8.0
"""
import os
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

print(f"Gradio version: {gr.__version__}")

# ── Load API key ──
load_dotenv("groq_api.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATA_FOLDER  = os.getenv("DATA_FOLDER", "./data")
PORT         = int(os.getenv("PORT", 7860))

if not GROQ_API_KEY:
    print("⚠️  GROQ_API_KEY not found.")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ─────────────────────────────────────────────
# LOAD DOCUMENTS
# ─────────────────────────────────────────────
def load_documents(folder: str) -> list:
    docs     = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    path     = Path(folder)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 data/ folder created.")
        return docs

    for filepath in sorted(path.iterdir()):
        text = ""

        if filepath.suffix.lower() == ".txt":
            for enc in ["utf-8", "utf-16", "cp1252", "latin-1"]:
                try:
                    text = filepath.read_text(encoding=enc, errors="ignore")
                    break
                except Exception:
                    continue

        elif filepath.suffix.lower() == ".pdf":
            try:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                print("⚠️  pdfplumber not installed.")
            except Exception as e:
                print(f"⚠️  Could not read {filepath.name}: {e}")

        if not text.strip():
            continue

        for chunk in splitter.split_text(text):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": filepath.name}
            ))
        print(f"  ✔ Loaded: {filepath.name}")

    print(f"\n✅ {len(docs)} chunks from {len(set(d.metadata['source'] for d in docs))} file(s)")
    return docs

# ─────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────
print("\n⏳ Loading embedding model…")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
docs = load_documents(DATA_FOLDER)

if docs:
    db = Chroma.from_documents(
        docs, embeddings,
        collection_name="niche_db",
        persist_directory="./chroma_store"
    )
else:
    db = Chroma(
        embedding_function=embeddings,
        collection_name="niche_db",
        persist_directory="./chroma_store"
    )
    print("⚠️  No documents found.")

print("✅ Ready!\n")

# ─────────────────────────────────────────────
# GENERATE RESPONSE
# ─────────────────────────────────────────────
num_files  = len(set(d.metadata["source"] for d in docs)) if docs else 0
num_chunks = len(docs)

def chat(message: str, history: list) -> str:
    if not message.strip():
        return ""
    if not client:
        return "❌ No GROQ_API_KEY found."

    context_text = ""
    sources = []
    if docs:
        results      = db.similarity_search(message, k=3)
        context_text = "\n\n".join(r.page_content for r in results)
        sources      = list(dict.fromkeys(
            r.metadata.get("source", "") for r in results
            if r.metadata.get("source")
        ))

    system = (
        "You are a helpful assistant called Niche GPT. "
        "Answer using ONLY the context below. Be concise — 2 to 4 sentences. "
        "If the answer is not in the context, say so honestly.\n\n"
        f"--- CONTEXT ---\n{context_text}\n--- END CONTEXT ---"
        if context_text else
        "You are a helpful assistant called Niche GPT. Answer clearly and concisely."
    )

    messages = [{"role": "system", "content": system}]
    for item in history[-6:]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]: messages.append({"role": "user",      "content": str(item[0])})
            if item[1]: messages.append({"role": "assistant", "content": str(item[1])})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"❌ Groq API error: {e}"

    if sources:
        answer += f"\n\n📄 *Sources: {', '.join(sources)}*"

    return answer

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=chat,
    title="🤖 Niche GPT",
    description=(
        f"Your Documents · Groq · Llama 3.1 8B · Free | "
        f"📚 {num_chunks} chunks · {num_files} file(s) | "
        f"🔍 MiniLM-L6 Embeddings"
    ),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)