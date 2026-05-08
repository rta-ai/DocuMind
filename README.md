# 🧠 DocuMind
### RAG-powered PDF Chat Assistant

Built with **Groq + LLaMA 3.3 + ChromaDB + Streamlit**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Live Demo
👉 [DocuMind on Hugging Face](https://huggingface.co/spaces/RTAAI/DocuMind)

## ✨ Features
- 📄 Upload any PDF (up to 200MB)
- 💬 Q&A — ask anything about your document
- 📋 Summarize — full or 5-point bullet summary
- 🔍 Key Extract — dates, names, figures, terms
- ⚡ Streaming responses via Groq API
- 🧠 RAG pipeline with ChromaDB + LLaMA 3.3

## 🛠️ Tech Stack
| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector DB | ChromaDB |
| LLM | Groq API — llama-3.3-70b-versatile |

## ⚙️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/DocuMind.git
cd DocuMind
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
Create a `.env` file: