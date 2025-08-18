import streamlit as st
from rag import answer

st.set_page_config(page_title="🧵 Fashion AI Research Copilot", layout="wide")
st.title("🧵 Fashion AI Research Copilot (RAG)")
st.caption("Ask about SGDiff, ControlNet, IP-Adapter, datasets, training setups — grounded in your PDFs.")

if "history" not in st.session_state:
    st.session_state.history = []

q = st.chat_input("Ask a question about your fashion/GenAI PDFs...")
if q:
    with st.spinner("Searching your library and drafting an answer..."):
        a, cites = answer(q)
    st.session_state.history.append(("user", q))
    suffix = f"\n\n**Sources:** {', '.join(cites)}" if cites else ""
    st.session_state.history.append(("assistant", a + suffix))

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
