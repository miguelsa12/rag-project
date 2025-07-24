# app.py
import streamlit as st
from rag_pipeline import RAGPipeline

# RAGPipeline
@st.cache_resource
def load_rag():
    return RAGPipeline()

rag = load_rag()

# Load & process data
if 'documents' not in st.session_state:
    st.session_state.documents = rag.load_pdfs()
    st.session_state.embedded_chunks = rag.embed_chunks(st.session_state.documents)

st.title("📄💬 RAG Demo: Ask Your PDFs")

question = st.text_input("Type your question here:")

if st.button("Generate Answer") and question:
    relevant_chunks = rag.retrieve_relevant_chunks(question, st.session_state.embedded_chunks)
    answer = rag.generate_answer(question, relevant_chunks)

    # Evaluación
    similarity_score = rag.evaluate_answer(question, answer)

    # Mostrar resultado
    st.subheader("🔍 Answer")
    st.write(answer)

    st.metric(label="🧪 Similarity Score (Q-A)", value=f"{similarity_score:.2f}")

    st.subheader("📚 Context used")
    for i, chunk in enumerate(relevant_chunks, 1):
        st.markdown(f"**Chunk {i}:** {chunk['text']}")
