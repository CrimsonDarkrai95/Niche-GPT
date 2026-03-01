"""
Niche GPT — RAG Chatbot
Users upload their own PDFs/TXTs per session.
Uses Groq (free Llama 3) + ChromaDB (in-memory) + Gradio 6.8.0
"""
import os
import uuid
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
PORT         = int(os.getenv("PORT", 7860))

if not GROQ_API_KEY:
    print("⚠️  GROQ_API_KEY not found.")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── Settings ──
MAX_FILES       = 25    # max files per session
MAX_FILE_MB     = 200   # max total MB per session
MAX_PER_FILE_MB = 20    # max MB per individual file
CHUNK_SIZE      = 400   # smaller = more precise retrieval
CHUNK_OVERLAP   = 80    # higher overlap = no context lost at boundaries
BATCH_SIZE      = 64    # full speed, safe on 16GB RAM

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

print("\n⏳ Loading embedding model…")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": BATCH_SIZE},
)
print("✅ Ready!\n")

# ─────────────────────────────────────────────
# DYNAMIC K — scales with document volume
# ─────────────────────────────────────────────
def get_k(num_chunks: int) -> int:
    if num_chunks < 100:    # short file, ~5 pages
        return 4
    elif num_chunks < 500:  # medium, ~20 pages
        return 6
    elif num_chunks < 1000: # long, ~50 pages
        return 8
    else:                   # very long / many files
        return 10

# ─────────────────────────────────────────────
# PROCESS UPLOADED FILES
# ─────────────────────────────────────────────
def process_files(files, session_state):
    if not files:
        return session_state, "⚠️ No files selected."

    docs         = []
    newly_loaded = []
    skipped      = []
    errors       = []

    # ── Check session file limit ──
    remaining_slots = MAX_FILES - len(session_state["sources"])
    if remaining_slots <= 0:
        return session_state, f"❌ File limit reached ({MAX_FILES} files max per session). Start a new session to upload more."

    # ── Check total size limit ──
    current_mb   = session_state["total_mb"]
    new_files_mb = sum(Path(f.name).stat().st_size for f in files) / (1024 * 1024)
    if current_mb + new_files_mb > MAX_FILE_MB:
        return session_state, (
            f"❌ Upload would exceed {MAX_FILE_MB}MB session limit.\n"
            f"Currently using {current_mb:.1f}MB. "
            f"These files are {new_files_mb:.1f}MB."
        )

    for file in files:
        filepath = Path(file.name)

        # ── Respect remaining slot limit ──
        if len(newly_loaded) >= remaining_slots:
            errors.append(f"⚠️ Skipped {filepath.name} — session file limit ({MAX_FILES}) reached.")
            break

        # ── Skip already-processed files ──
        if filepath.name in session_state["sources"]:
            skipped.append(filepath.name)
            continue

        # ── Per-file size check ──
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_PER_FILE_MB:
            errors.append(f"❌ {filepath.name} is {file_size_mb:.1f}MB — exceeds {MAX_PER_FILE_MB}MB per-file limit.")
            continue

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
            except Exception as e:
                errors.append(f"❌ Could not read {filepath.name}: {e}")
                continue

        if not text.strip():
            errors.append(f"⚠️ {filepath.name} appears empty or unreadable.")
            continue

        for chunk in splitter.split_text(text):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": filepath.name}
            ))
        newly_loaded.append(filepath.name)

    # ── Build status message ──
    status_parts = []

    if skipped:
        status_parts.append(f"⏭️ Already loaded (skipped): {', '.join(skipped)}")
    if errors:
        status_parts.extend(errors)

    if not docs and not skipped:
        return session_state, "\n".join(status_parts) or "❌ No readable content found."

    if docs:
        if session_state["db"] is None:
            db = Chroma.from_documents(
                docs, embeddings,
                collection_name=f"s_{uuid.uuid4().hex[:8]}",
            )
            session_state["db"] = db
        else:
            session_state["db"].add_documents(docs)

        session_state["sources"]    = list(set(session_state["sources"] + newly_loaded))
        session_state["num_chunks"] += len(docs)
        session_state["total_mb"]   += new_files_mb

        slots_left  = MAX_FILES - len(session_state["sources"])
        current_k   = get_k(session_state["num_chunks"])
        status_parts.append(
            f"✅ Added {len(docs)} chunks from {len(newly_loaded)} file(s): {', '.join(newly_loaded)}"
        )
        status_parts.append(
            f"\n📚 Total: {session_state['num_chunks']} chunks · "
            f"{session_state['total_mb']:.1f}MB used · "
            f"{slots_left}/{MAX_FILES} slots remaining · "
            f"retrieval K={current_k}\n"
            + "\n".join(f"  • {s}" for s in session_state["sources"])
        )
    else:
        status_parts.append("✅ No new files to process.")

    return session_state, "\n".join(status_parts)

# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────
def chat(message, history, session_state):
    if not message.strip():
        return history, ""
    if not client:
        return history + [{"role": "user", "content": message},
                          {"role": "assistant", "content": "❌ No GROQ_API_KEY found."}], ""
    if session_state["db"] is None:
        return history + [{"role": "user", "content": message},
                          {"role": "assistant", "content": "⚠️ Please upload and process at least one file before chatting."}], ""

    # ── Dynamic K based on current chunk count ──
    k            = get_k(session_state["num_chunks"])
    results      = session_state["db"].similarity_search(message, k=k)
    context_text = "\n\n".join(r.page_content for r in results)
    sources      = list(dict.fromkeys(
        r.metadata.get("source", "") for r in results
        if r.metadata.get("source")
    ))

    system = (
        "You are a helpful study assistant called Niche GPT. "
        "Answer using ONLY the context below. Be thorough but precise. "
        "If multiple questions are asked, answer each one clearly. "
        "If the answer is not in the context, say so honestly. Do not make things up.\n\n"
        f"--- CONTEXT ---\n{context_text}\n--- END CONTEXT ---"
    )

    messages = [{"role": "system", "content": system}]
    for item in history[-6:]:
        if isinstance(item, dict) and "role" in item and "content" in item:
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"❌ Groq API error: {e}"

    if sources:
        answer += f"\n\n📄 *Sources: {', '.join(sources)}*"

    return history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ], ""

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
def new_session():
    return {"db": None, "sources": [], "num_chunks": 0, "total_mb": 0.0}

with gr.Blocks(title="Niche GPT") as demo:

    session_state = gr.State(new_session())

    gr.HTML(f"""
    <div style="text-align:center; padding:30px 20px 10px; font-family:sans-serif;">
        <h1 style="font-size:2rem; font-weight:700; margin:0;">🤖 Niche GPT</h1>
        <p style="color:#666; margin-top:6px; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em;">
            Upload · Ask · Learn
        </p>
        <hr style="width:60px; border:2px solid #7c6af7; margin:12px auto;">
        <p style="color:#999; font-size:0.78rem;">
            🔒 Session-only · Max {MAX_FILES} files · Max {MAX_FILE_MB}MB · Nothing stored permanently
        </p>
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1, min_width=280):
            gr.Markdown(f"### 📁 Upload Documents\n*Max {MAX_FILES} files · {MAX_PER_FILE_MB}MB per file · {MAX_FILE_MB}MB per session*")
            file_upload = gr.File(
                label="PDF or TXT files",
                file_types=[".pdf", ".txt"],
                file_count="multiple",
            )
            process_btn = gr.Button("⚡ Process Files", variant="primary", size="lg")
            status_box  = gr.Textbox(
                label="Status",
                lines=8,
                interactive=False,
                placeholder="Upload files then click Process Files...",
            )

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with your Documents")
            chatbot = gr.Chatbot(height=420, show_label=False)
            with gr.Row():
                msg_box  = gr.Textbox(
                    placeholder="Ask one or more questions about your documents…",
                    show_label=False,
                    scale=8,
                )
                send_btn  = gr.Button("Send →", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

    process_btn.click(
        fn=process_files,
        inputs=[file_upload, session_state],
        outputs=[session_state, status_box],
    )
    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, session_state],
        outputs=[chatbot, msg_box],
    )
    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot, session_state],
        outputs=[chatbot, msg_box],
    )
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, msg_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)