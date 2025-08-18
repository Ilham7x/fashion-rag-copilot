# import os
# from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate

# load_dotenv()

# INDEX_DIR = "chroma_index"
# EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # same as build_index.py

# # --- LLM selection ---
# USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
# if USE_OPENAI:
#     from langchain_openai import ChatOpenAI
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# else:
#     class DummyLLM:
#         def invoke(self, messages):
#             content = messages[-1].content
#             return type("R", (), {"content": "🔎 LLM disabled. Showing retrieved CONTEXT only:\n\n" + content[:2000]})
#     llm = DummyLLM()

# # Embeddings + Vector DB
# emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# db = Chroma(persist_directory=INDEX_DIR, embedding_function=emb)
# retriever = db.as_retriever(search_kwargs={"k": 5})

# SYSTEM = """You are a careful assistant for fashion & generative AI.
# Use ONLY the provided CONTEXT to answer. If the answer is not in the context,
# say "I don't know from the given sources." Cite as [source: filename p.X]. Be concise."""
# PROMPT = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM),
#     ("human", "Question: {q}\n\nCONTEXT:\n{context}")
# ])

# def _format_citations(docs):
#     seen, cites = set(), []
#     for d in docs:
#         tag = f"{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}"
#         if tag not in seen:
#             seen.add(tag); cites.append(tag)
#     return cites

# def answer(q: str):
#     docs = retriever.get_relevant_documents(q)
#     context = "\n\n".join(
#         f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}] {d.page_content}"
#         for d in docs
#     )
#     msgs = PROMPT.format_messages(q=q, context=context)
#     res = llm.invoke(msgs)
#     cites = _format_citations(docs)
#     return res.content, cites
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient  # <-- direct HF client

load_dotenv()

INDEX_DIR = "chroma_index"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

HF_REPO = os.getenv("HF_REPO_ID", "HuggingFaceH4/zephyr-7b-beta")

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN in .env")

client = InferenceClient(model=HF_REPO, token=HF_TOKEN)

# retrieval setup
emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=INDEX_DIR, embedding_function=emb)
retriever = db.as_retriever(search_kwargs={"k": 5})

SYSTEM = (
    "You are a careful assistant for fashion & generative AI.\n"
    "Use ONLY the provided CONTEXT to answer. If the answer is not in the context,\n"
    "say \"I don't know from the given sources.\" Cite as [source: filename p.X]. Be concise."
)

def _format_citations(docs):
    seen, cites = set(), []
    for d in docs:
        tag = f"{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}"
        if tag not in seen:
            seen.add(tag); cites.append(tag)
    return cites

def _chat(messages, max_new_tokens=350, temperature=0.2, top_p=0.9):
    """Call HF chat endpoint directly and return text."""
    resp = client.chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return resp.choices[0].message.content

def answer(q: str):
    docs = retriever.get_relevant_documents(q)
    context = "\n\n".join(
        f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}] {d.page_content}"
        for d in docs
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {q}\n\nCONTEXT:\n{context}\n\nAnswer:"},
    ]
    text = _chat(messages)
    cites = _format_citations(docs)
    return text, cites
