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
├── data/ # Raw PDF files
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

## 📥 Build the Index

Place your PDFs into `data/` and run:

```bash
python src/build_index.py
```

This will:
- Load all PDFs from data/
- Split them into chunks (500 tokens with overlap)
- Clean text and validate encodings
- Embed chunks with all-mpnet-base-v2
- Save them into a persistent Chroma index

## 🔎 Query the Index
Once the index is built, run the RAG pipeline:
```bash
python src/rag.py "What does the FashionSD-X paper propose?"
```
The system retrieves relevant chunks and returns an answer based on them.

## 📊 Example Workflow
1. Add fashion-related papers (e.g., FashionSD-X, SGDiff, StyleGAN).
2. Build the index with build_index.py.
3. Ask domain-specific questions with rag.py.
4. (Optional) Run app.py to expose a simple API / UI for querying.

## 🚀 Future Improvements
- Add re-ranking for better retrieval quality.
- Integrate OpenAI / Llama-2 / other LLMs for more fluent generation.
- Build a frontend dashboard for interactive querying.

## 📜 License
MIT License
