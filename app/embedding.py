import pandas as pd
import numpy as np
import faiss
import re

def load_clinvec(path="app/data/ClinVec_phecode.csv", meta_path="app/data/ClinGraph_nodes.csv"):
    df = pd.read_csv(path)
    node_df = pd.read_csv(meta_path, sep="\t")
    df['node_index'] = df.index
    full_df = df.merge(node_df, how="inner", on="node_index")
    emb_mat = df.drop(columns=["node_index"]).values.astype("float32")
    return emb_mat, full_df

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def extract_codes_from_text(text):
    return re.findall(r"PheCode:\w+", text)

def get_clinvec_context(question, faiss_index, emb_df, emb_matrix, k=3):
    codes = extract_codes_from_text(question)
    if not codes:
        return ""
    matches = []
    for code in codes:
        candidates = emb_df[emb_df["code"] == code]
        if not candidates.empty:
            row_idx = candidates["node_index"].values[0]
            emb_vec = emb_matrix[row_idx].reshape(1, -1)
            D, I = faiss_index.search(emb_vec, k)
            matched = emb_df.iloc[I[0]][["code", "name"]]
            matches.append(matched.to_dict(orient="records"))
    if not matches:
        return "No similar codes found."
    return "\n".join([f"{m['code']}: {m['name']}" for group in matches for m in group])

from sentence_transformers import SentenceTransformer

text_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_text_index(chunks):
    embeddings = text_embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index, embeddings, chunks

def get_top_chunks(question, index, chunks, k=3):
    q_vec = text_embedder.encode([question], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_vec, k)
    return "\n".join([chunks[i] for i in I[0]])
