"""
Niche GPT — RAG Chatbot
Uses Groq (free Llama 3) + ChromaDB + Gradio 6
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

# ── Load API key ──
load_dotenv("groq_api.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATA_FOLDER  = os.getenv("DATA_FOLDER", "./data")

if not GROQ_API_KEY:
    print("⚠️  GROQ_API_KEY not found. Check your groq_api.env file.")

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
        print(f"📁 data/ folder created. Add your .txt or .pdf files there.")
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
                print("⚠️  pdfplumber not installed. Run: pip install pdfplumber")
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
        return "❌ No GROQ_API_KEY found. Check your groq_api.env file."

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
    for human, bot in history[-6:]:
        if human: messages.append({"role": "user",      "content": human})
        if bot:   messages.append({"role": "assistant", "content": bot})
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
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Dark background only on outer containers */
body, .gradio-container {
    background: #0a0a0f !important;
}

footer, .built-with { display: none !important; }

/* Chatbox */
#chatbox {
    background: #111118 !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 14px !important;
}

/* Chat bubbles */
.message {
    background: #1a1a2e !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin: 6px 0 !important;
    color: #e8e8f0 !important;
}
.message p, .message span, .message li,
.message code, .message pre,
.bot, .bot p, .human, .human p {
    color: #1a1a2e !important;
}
/* Override — actual message text should be dark on light bubble */
.message.bot { background: #f0f0f5 !important; }
.message.human { background: #e8e4ff !important; }
.message.bot p, .message.bot span,
.message.human p, .message.human span {
    color: #111111 !important;
}

/* Send button */
#send-btn {
    background: linear-gradient(135deg, #7c6af7, #4fd1c5) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
}
#send-btn:hover { opacity: 0.85 !important; }

/* Clear button */
#clear-btn {
    border-radius: 10px !important;
    font-size: 0.85rem !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #1e1e2e; border-radius: 4px; }
"""

num_files  = len(set(d.metadata["source"] for d in docs)) if docs else 0
num_chunks = len(docs)

def chat(message: str, history: list):
    if not message.strip():
        return history, ""
    response = generate_response(message, history)
    return history + [(message, response)], ""

with gr.Blocks(title="Niche GPT", css=CSS) as demo:

    gr.HTML(f"""
    <div style="text-align:center; padding:36px 20px 20px; background:#0a0a0f; font-family:'DM Sans',sans-serif;">
        <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:700; letter-spacing:-0.03em;
                    background:linear-gradient(135deg,#ffffff 30%,#4fd1c5 100%);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
            Niche GPT
        </div>
        <div style="font-size:0.78rem; color:#5a5a70; margin-top:6px; letter-spacing:0.08em; text-transform:uppercase;">
            Your Documents · Powered by Groq · Llama 3 8B
        </div>
        <div style="width:60px; height:2px; background:linear-gradient(90deg,#7c6af7,#4fd1c5); margin:14px auto 20px; border-radius:2px;"></div>
        <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
            <span style="font-size:0.73rem; color:#5a5a70; background:#111118; border:1px solid #1e1e2e; border-radius:20px; padding:4px 13px;">
                📚 <b style="color:#4fd1c5">{num_chunks}</b> chunks · <b style="color:#4fd1c5">{num_files}</b> file(s)
            </span>
            <span style="font-size:0.73rem; color:#5a5a70; background:#111118; border:1px solid #1e1e2e; border-radius:20px; padding:4px 13px;">
                ⚡ Groq <b style="color:#4fd1c5">Llama 3 8B</b> · Free
            </span>
            <span style="font-size:0.73rem; color:#5a5a70; background:#111118; border:1px solid #1e1e2e; border-radius:20px; padding:4px 13px;">
                🔍 Embeddings: <b style="color:#4fd1c5">MiniLM-L6</b>
            </span>
        </div>
    </div>
    """)

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbox",
        height=460,
        show_label=False,
        avatar_images=(None, None),
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask something about your documents…",
            show_label=False,
            elem_id="msg-box",
            lines=1,
            max_lines=6,
            scale=8,
        )
        send_btn  = gr.Button("Send →",  elem_id="send-btn",  scale=1, min_width=90)
        clear_btn = gr.Button("Clear",   elem_id="clear-btn", scale=1, min_width=70)

    msg.submit(chat,  [msg, chatbot], [chatbot, msg])
    send_btn.click(chat,  [msg, chatbot], [chatbot, msg])
    clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)