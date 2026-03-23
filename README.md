# 🧠 DocMind — RAG Chatbot for Custom Documents

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Store-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/RAG-Architecture-blueviolet?style=for-the-badge"/>
</p>

> Upload any PDF document and chat with it using the power of **Retrieval-Augmented Generation (RAG)**. Supports multiple LLM providers — OpenAI, Google Gemini, and Groq.

---

## 🚀 Demo

![DocMind Demo](assets/demo.gif)

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PyPDFLoader → Text Extraction
    │
    ▼
RecursiveCharacterTextSplitter → Chunking (1000 tokens, 150 overlap)
    │
    ▼
Embeddings (OpenAI / HuggingFace) → FAISS Vector Store
    │
    ▼
User Query → Similarity Search (Top-4 chunks)
    │
    ▼
LLM (GPT-4 / Gemini / LLaMA3) + RAG Prompt → Answer
```

---

## ✨ Features

- 📄 Upload any PDF document
- 🤖 Choose your LLM — OpenAI GPT-4, Google Gemini, or Groq (LLaMA3)
- ⚡ Fast semantic search using FAISS vector store
- 🔗 LangChain-powered RetrievalQA pipeline
- 💬 Clean chat interface built with Streamlit
- 🔒 API keys entered securely at runtime — never stored

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM Providers | OpenAI GPT-4o-mini, Google Gemini 1.5 Flash, Groq LLaMA3 |
| Embeddings | OpenAI Ada-002, Google Embedding-001, HuggingFace MiniLM |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| RAG Framework | LangChain |
| PDF Loader | PyPDFLoader |
| UI | Streamlit |

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/chhabralovish/docmind-rag-chatbot.git
cd docmind-rag-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your API key
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🔑 API Keys

Get your free API keys here:

| Provider | Link | Free Tier |
|---|---|---|
| OpenAI | [platform.openai.com](https://platform.openai.com/) | $5 free credits |
| Google Gemini | [aistudio.google.com](https://aistudio.google.com/) | ✅ Free |
| Groq | [console.groq.com](https://console.groq.com/) | ✅ Free & Fast |

---

## 📁 Project Structure

```
docmind-rag-chatbot/
│
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # Core RAG logic (load, chunk, embed, query)
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore              # Python gitignore
└── README.md               # This file
```

---

## 🧠 How RAG Works

**RAG (Retrieval-Augmented Generation)** combines the power of vector search with LLMs:

1. **Indexing** — The PDF is split into chunks and converted into vector embeddings stored in FAISS
2. **Retrieval** — When you ask a question, the top-4 most semantically similar chunks are retrieved
3. **Generation** — The LLM uses those chunks as context to generate a grounded, accurate answer

This prevents hallucinations and keeps the LLM's answers strictly based on your document.

---

## 👨‍💻 Author

**Lovish Chhabra** — Data Scientist & AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/lovish-chhabra/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/chhabralovish)

---

## 📄 License

MIT License — feel free to use, modify and distribute.
