"""
Niche GPT — RAG Chatbot
Uses Groq (free Llama 3) + ChromaDB + Gradio 4.44.0
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
PORT         = int(os.getenv("PORT", 10000))

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
def generate_response(query: str, history: list) -> str:
    if not query.strip():
        return ""
    if not client:
        return "❌ No GROQ_API_KEY found."

    context_text = ""
    sources = []
    if docs:
        results      = db.similarity_search(query, k=3)
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
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]: messages.append({"role": "user",      "content": item[0]})
            if item[1]: messages.append({"role": "assistant", "content": item[1]})
    messages.append({"role": "user", "content": query})

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
num_files  = len(set(d.metadata["source"] for d in docs)) if docs else 0
num_chunks = len(docs)

def chat(message: str, history: list):
    if not message.strip():
        return history, ""
    response = generate_response(message, history)
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": response},
    ]
    return history, ""

with gr.Blocks(title="Niche GPT") as demo:

    gr.HTML(f"""
    <div style="text-align:center; padding:36px 20px 20px; font-family:sans-serif;">
        <h1 style="font-size:2rem; font-weight:700; margin:0;">🤖 Niche GPT</h1>
        <p style="color:#666; margin-top:6px; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em;">
            Your Documents · Groq · Llama 3.1 8B · Free
        </p>
        <hr style="width:60px; border:2px solid #7c6af7; margin:14px auto;">
        <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:10px;">
            <span style="font-size:0.75rem; background:#f0f0f5; border-radius:20px; padding:4px 12px;">
                📚 <b>{num_chunks}</b> chunks · <b>{num_files}</b> file(s)
            </span>
            <span style="font-size:0.75rem; background:#f0f0f5; border-radius:20px; padding:4px 12px;">
                ⚡ Llama 3.1 8B · Free
            </span>
            <span style="font-size:0.75rem; background:#f0f0f5; border-radius:20px; padding:4px 12px;">
                🔍 MiniLM-L6 Embeddings
            </span>
        </div>
    </div>
    """)

    chatbot = gr.Chatbot(
        [],
        height=460,
        show_label=False,
        type="messages",
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask something about your documents…",
            show_label=False,
            lines=1,
            max_lines=6,
            scale=8,
        )
        send_btn  = gr.Button("Send →", scale=1, min_width=90, variant="primary")
        clear_btn = gr.Button("Clear",  scale=1, min_width=70)

    msg.submit(chat,  [msg, chatbot], [chatbot, msg])
    send_btn.click(chat,  [msg, chatbot], [chatbot, msg])
    clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)