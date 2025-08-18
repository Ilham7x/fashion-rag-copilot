import glob, os, re
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_DIR = "chroma_index"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  

def load_pdfs():
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        loader = PyPDFLoader(path)
        pages = loader.load()
        for d in pages:
            d.metadata.update({"source": os.path.basename(path)})
        docs += pages
    return docs

def clean(text) -> str:
    try:
        text = "" if text is None else str(text)
    except Exception:
        return ""
    text = text.replace("\x00", " ")                    # remove nulls
    text = re.sub(r"[^\S\r\n]+", " ", text)             # collapse whitespace (keep newlines)
    text = text.strip()
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\r\n\t")
    return text

def probe_bad_texts(texts: List[str], model_name: str) -> List[int]:
    """Return indices of texts that fail tokenizer/encoder."""
    st = SentenceTransformer(model_name)
    bad_indices = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or t == "":
            bad_indices.append(i)
            continue
        try:
            # encode a single-item batch; fast + faithful to real call
            st.encode([t], show_progress_bar=False, normalize_embeddings=False)
        except Exception as e:
            print(f"[probe] Bad text at idx {i} (len={len(t)}): {type(t)} -> {e}")
            print("[probe] snippet:", repr(t[:200]))
            bad_indices.append(i)
    return bad_indices

def main():
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages from PDFs in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    texts, metadatas = [], []
    for d in chunks:
        t = clean(d.page_content)
        if not t:
            continue
        texts.append(t)
        metadatas.append({
            "source": d.metadata.get("source", "?"),
            "page": d.metadata.get("page", "?"),
        })

    print(f"Prepared {len(texts)} cleaned chunks. Probing for tokenizer issues...")
    bad_idxs = probe_bad_texts(texts, EMBED_MODEL)
    if bad_idxs:
        print(f"Found {len(bad_idxs)} problematic chunks. Skipping them.")
        keep_texts, keep_meta = [], []
        bad_set = set(bad_idxs)
        for i, (t, m) in enumerate(zip(texts, metadatas)):
            if i in bad_set:
                continue
            keep_texts.append(t)
            keep_meta.append(m)
        texts, metadatas = keep_texts, keep_meta

    print(f"Final chunk count going to embeddings: {len(texts)}")

    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma.from_texts(
        texts=texts,
        embedding=emb,
        metadatas=metadatas,
        persist_directory=INDEX_DIR
    )
    db.persist()
    print(f"✅ Indexed {len(texts)} chunks and saved to {INDEX_DIR}")

if __name__ == "__main__":
    main()
