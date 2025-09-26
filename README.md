# Fahion RAG Copilot
An AI-powered research assistant for fashion &amp; GenAI literature using RAG (Retrieval-Augmented Generation)

---

## 📌 Project Overview

This project demonstrates how to combine **LangChain**, **Hugging Face embeddings**, and a **Chroma vector store** into a working RAG pipeline.  
- PDFs (research papers, datasets, reports) are ingested and indexed.  
- A user can query the index, and the system retrieves the most relevant passages.  
- The retrieved chunks are provided to a language model, enabling accurate, citation-backed answers.  

The long-term aim is to serve as a **research copilot** for fashion AI, with a focus on works such as **FashionSD-X** and multimodal garment generation.

---

## 🗂 Project Structure
```
fashion-rag-copilot/
│
├── data/ # Raw PDF files (not committed to GitHub)
├── chroma_index/ # Persisted ChromaDB index
│
├── src/
│ ├── build_index.py # Preprocess PDFs and build the vector index
│ ├── rag.py # RAG pipeline (query engine)
│ └── app.py # (Optional) Simple interface / FastAPI app
│
├── .gitignore # Ignore PDFs, env files, large binaries
├── README.md # Project documentation
├── requirements.txt # Dependencies
└── LICENSE
```
---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Ilham7x/fashion-rag-copilot.git
cd fashion-rag-copilot
```
2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
