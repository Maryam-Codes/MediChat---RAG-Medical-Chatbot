# 🩺 MediChat - Evidence-Based Medical AI Assistant (RAG)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-yellow.svg)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Inference-Groq_LPU-orange.svg)](https://groq.com/)

**MediChat** is an intelligent, retrieval-augmented healthcare chatbot designed to deliver authentic, hallucination-free medical information. By pairing **The Gale Encyclopedia of Medicine** with high-speed **Groq LPU inference** and **FAISS vector search**, MediChat provides evidence-backed clinical explanations accompanied by direct page citations.

---

## 🌟 Key Features

* **🛡️ Zero Medical Hallucinations**: Constrained by strict prompt engineering to answer exclusively from peer-reviewed encyclopedia context.
* **📖 Verified Citations**: Every generated response includes reference chunk details and original book page numbers.
* **⚡ Sub-Second Inference**: Powered by Groq LPU technology for near-instant responses.
* **🎨 Modern Health-Tech UI**: Beautiful dark-mode glassmorphism interface built with Streamlit, including quick question chips and source expanders.
* **👋 Conversational Etiquette**: Gracefully handles natural greetings while reserving medical rigor for clinical questions.

---

## 🏗️ Architecture & Pipeline

```text
  [ Medical PDF ] 
         │
         ▼
[ RecursiveTextSplitter (500 chunks / 50 overlap) ]
         │
         ▼
[ HuggingFace Embeddings (all-MiniLM-L6-v2) ]
         │
         ▼
[ FAISS Vector Store ] ──(Top-3 Semantic Search)──┐
                                                 │
                                                 ▼
[ User Query ] ──► [ Custom Medical Prompt ] ──► [ Groq LPU (LLM) ] ──► [ MediChat Response + Citations ]
```

---

## 🛠️ Tech Stack

* **Language**: Python 3.10+
* **LLM Engine**: Groq LPU (`openai/gpt-oss-20b`)
* **Vector Store**: FAISS (Facebook AI Similarity Search)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
* **RAG Framework**: LangChain
* **Frontend UI**: Streamlit (Custom Glassmorphism CSS)

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MediChat.git
cd MediChat
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your free Groq API Key in `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Get a free key from [console.groq.com](https://console.groq.com/keys))*

### 4. (Optional) Rebuild the Vector Store
The pre-computed vector database is included in `vectorstore/db_faiss/`. If you wish to re-index the PDF from scratch:
```bash
python create_memory_for_llm.py
```

### 5. Launch the Application
```bash
python -m streamlit run medichat.py
```
Open your browser at `http://localhost:8501`.

---

## ⚠️ Disclaimer
*MediChat is an AI retrieval tool created for educational and research demonstration purposes. It does not replace professional medical advice, clinical diagnosis, or hospital treatment. Always consult a qualified medical professional for health conditions.*

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
