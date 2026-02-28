# 🤖 Niche GPT

> A Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own documents — powered by **Groq's free Llama 3.1 API**, **ChromaDB**, and **Gradio**.

---

## ✨ Features

- 📚 **RAG Pipeline** — Searches your documents and retrieves the most relevant context before answering
- ⚡ **Groq + Llama 3.1 8B** — Fast, intelligent responses using a free API (no GPU, no paid tier)
- 🔍 **ONNX Embeddings** — Lightweight local embeddings, no PyTorch required
- 📄 **PDF + TXT Support** — Drop any `.pdf` or `.txt` files into the `data/` folder
- 💬 **Chat History** — Remembers the last 6 turns of conversation for context
- 📌 **Source Citations** — Every answer shows which file it came from
- 🎨 **Clean Dark UI** — Minimalist Gradio interface with a dark theme

---

## 🧠 How It Works

```
Your Question
     │
     ▼
ONNX MiniLM Embeddings (runs locally)
     │
     ▼
ChromaDB finds top 3 most relevant chunks from your documents
     │
     ▼
Groq API sends chunks + question to Llama 3.1 8B (free)
     │
     ▼
Clean, concise answer with source citations ✅
```

---

## 🗂️ Project Structure

```
Niche-GPT/
├── hello1.py           ← main app (RAG pipeline + Gradio UI)
├── requirements.txt    ← all dependencies
├── runtime.txt         ← pins Python 3.11 for deployment
├── .gitignore
├── data/               ← your .txt and .pdf documents go here
└── README.md
```

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/CrimsonDarkrai95/Niche-GPT.git
cd Niche-GPT
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get your free Groq API key
1. Sign up at **https://console.groq.com** (free, no credit card)
2. Go to **API Keys → Create API Key**
3. Copy the key (starts with `gsk_...`)

### 5. Create your API key file
Create a file named `groq_api.env` in the project folder:
```
GROQ_API_KEY=gsk_your_key_here
```

### 6. Add your documents
Drop `.txt` or `.pdf` files into the `data/` folder.

### 7. Run
```bash
python hello1.py
```

Open **http://localhost:7860** in your browser.

---

## ☁️ Deployment

This project is configured for **Railway** deployment.

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add environment variable: `GROQ_API_KEY` = your Groq key
4. Set start command: `python hello1.py`
5. Deploy ✅

> **Note:** The `groq_api.env` file is gitignored for security. On Railway, set `GROQ_API_KEY` as an environment variable in the dashboard instead.

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.1 8B via Groq API (free) |
| Embeddings | ONNX MiniLM-L6-v2 (local) |
| Vector Store | ChromaDB |
| Document Loader | LangChain + pdfplumber |
| UI | Gradio 3.50.2 |
| Language | Python 3.11 |

---

## ⚠️ Important Notes

- `groq_api.env` is **never committed** to GitHub — create it manually after cloning
- The `venv/` and `chroma_store/` folders are gitignored — recreate with `pip install -r requirements.txt`
- First run downloads the ONNX embedding model (~90MB, cached after that)

---

## 📄 License

MIT License — free to use, modify, and distribute.
