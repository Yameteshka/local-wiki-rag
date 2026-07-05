import streamlit as st
import requests
import sqlite3
import json
import numpy as np
from validation import ANNSearch, person4_index, sq8_store, corpus_f32

DB_PATH = "wikipedia.db"
ann = ANNSearch(person4_index, sq8_store, corpus_fallback=corpus_f32)

# get_embedding = функция от Человека 2, которая принимает строку и возвращает np.ndarray

def run_vector_search(query_text: str, top_k: int = 3):

    # Векторизация (Ожидаем код от Человека 2)
    # query_vector = get_embedding(query_text)

    # ВРЕМЕННАЯ ЗАГЛУШКА ВЕКТОРА (чтобы код не падал, пока Человек 2 не отдаст свою часть)
    query_vector = np.random.randn(256).astype(np.float32)

    ids, scores = ann.search(query_vector, top_k=top_k)

    results = []
    for doc_id, score in zip(ids, scores):
        results.append({
            "id": int(doc_id),
            "score": float(score)
        })

    max_score = float(scores[0]) if len(scores) > 0 else 0.0

    return {
        "max_score": max_score,
        "results": results
    }


def get_metadata_by_ids(vector_results):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    enriched_results = []

    for item in vector_results:
        cursor.execute("SELECT title, text FROM articles WHERE id = ?", (item["id"],))
        row = cursor.fetchone()
        if row:
            enriched_results.append({
                "title": row[0],
                "text": row[1],
                "score": item["score"]
            })
    conn.close()
    return enriched_results


def builtin_bm25_search(query, limit=3):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT id, title, text, bm25(articles) as bm25_score 
            FROM articles 
            WHERE articles MATCH ? 
            ORDER BY bm25_score 
            LIMIT ?
        """, (query, limit))
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        cursor.execute("SELECT id, title, text, 0 FROM articles WHERE text LIKE ? LIMIT ?", (f"%{query}%", limit))
        rows = cursor.fetchall()

    conn.close()

    return [{"title": r[1], "text": r[2], "score": abs(r[3])} for r in rows]


# Fallback logic

CONFIDENCE_THRESHOLD = 0.30


def retrieve_context(query):
    vector_data = run_vector_search(query)

    if vector_data["max_score"] < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low semantic confidence. Falling back to keyword search.", icon="⚠️")
        return builtin_bm25_search(query, limit=3)
    else:
        return get_metadata_by_ids(vector_data["results"])


# Integration with OLLAMA

def generate_answer(query, context_chunks):
    top_3 = context_chunks[:3]
    context_text = "\n\n".join(
        [f"[Source {i + 1}]: {chunk['title']}\n{chunk['text']}" for i, chunk in enumerate(top_3)]
    )

    system_prompt = f"""You are a helpful and precise assistant.
Answer the user's query IN ENGLISH ONLY, using strictly the facts provided in the Context below.
You must cite the sources using the exact format [Source X] at the end of the relevant sentences.
If the context does not contain the answer, say "I cannot answer this based on the provided local data."

Context:
{context_text}
"""

    payload = {
        "model": "gemma",
        "prompt": f"{system_prompt}\n\nUser Query: {query}",
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "Error generating response."), top_3
    except requests.exceptions.RequestException as e:
        return f"Ollama Connection Error: {e}", top_3


# UI interface streamlit

st.set_page_config(page_title="Local Wiki RAG", layout="wide")
st.title("📚 Local Wiki AI Search")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me about anything in the local Wikipedia..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching local databases..."):
            retrieved_chunks = retrieve_context(prompt)

            answer, used_sources = generate_answer(prompt, retrieved_chunks)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.markdown("### 📑 Sources Used")
            for i, source in enumerate(used_sources):
                with st.expander(f"[{i + 1}] {source['title']} (Score: {source['score']:.2f})"):
                    st.write(source['text'])
