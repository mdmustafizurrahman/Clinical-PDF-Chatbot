import streamlit as st
from app.pdf_reader import extract_text_from_pdf, chunk_text
from app.embedding import load_clinvec, build_faiss_index, get_clinvec_context
from app.embedding import build_text_index, get_top_chunks
from app.qa_engine import generate_answer
from app.evaluation_metrics import run_evaluation
import multiprocessing as mp
import time
import pandas as pd

mp.set_start_method("spawn", force=True)

st.set_page_config(page_title="Clinical PDF Chatbot", layout="wide")
st.title("Clinical PDF Chatbot")

@st.cache_resource
def load_resources():
    clinvec_emb, meta_df = load_clinvec()
    clinvec_index = build_faiss_index(clinvec_emb)
    return clinvec_emb, meta_df, clinvec_index

clinvec_emb, meta_df, clinvec_index = load_resources()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None
    st.session_state.pdf_chunks = []
if "metrics" not in st.session_state:
    st.session_state.metrics = []

uploaded_file = st.file_uploader("Upload a biomedical PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    index, _, stored_chunks = build_text_index(chunks)
    st.session_state.pdf_index = index
    st.session_state.pdf_chunks = stored_chunks
    st.success("PDF processed and indexed.")

user_input = st.chat_input("Ask a question from the document...")
if user_input and st.session_state.pdf_index:
    start_time = time.time()
    context = get_top_chunks(user_input, st.session_state.pdf_index, st.session_state.pdf_chunks)
    clinvec_context = get_clinvec_context(user_input, clinvec_index, meta_df, clinvec_emb)
    combined_context = context + "\n" + clinvec_context
    answer = generate_answer(user_input, combined_context)
    latency = round(time.time() - start_time, 2)
    answer_length = len(answer.split())

    # Run automated evaluation
    evaluation = run_evaluation(combined_context, user_input, answer)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.metrics.append({
        "question": user_input,
        "latency_sec": latency,
        "answer_length": answer_length,
        "faithfulness": evaluation["faithfulness_label"],
        "faithfulness_score": evaluation["faithfulness_score"],
        "relevance_score": evaluation["relevance_score"]
    })

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Show metrics
if st.session_state.metrics:
    st.markdown("### Business Metrics")
    for m in st.session_state.metrics[-5:]:
        st.write(
            f"Latency: {m['latency_sec']}s | "
            f"Length: {m['answer_length']} tokens | "
            f"Faithfulness: {m['faithfulness']} ({m['faithfulness_score']}) | "
            f"Relevance: {m['relevance_score']}"
        )

# Export metrics
if st.button("Export metrics as CSV"):
    df = pd.DataFrame(st.session_state.metrics)
    df.to_csv("chatbot_metrics.csv", index=False)
    st.success("Metrics saved to chatbot_metrics.csv")
