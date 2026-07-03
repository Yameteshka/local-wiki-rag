import streamlit as st
import requests
import sqlite3
import json

DB_PATH = "wikipedia.db"


# Человек 5 должен вернуть структуру вида:
# {
#     "max_score": 0.58,
#     "results": [
#         {"id": 42, "score": 0.58},
#         {"id": 112, "score": 0.51},
#         {"id": 7, "score": 0.49}
#     ]
# }

# Функция для извлечения текстов из БД по ID (Для векторного поиска Человека 5)
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


# Встроенный текстовый поиск BM25 через SQLite (Задача 2 - Fallback)
def builtin_bm25_search(query, limit=3):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # В SQLite FTS5 функция bm25() возвращает отрицательные значения (чем меньше/отрицательнее, тем лучше)
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
        # На случай, если пользователь ввел спецсимволы, которые сломали синтаксис MATCH
        cursor.execute("SELECT id, title, text, 0 FROM articles WHERE text LIKE ? LIMIT ?", (f"%{query}%", limit))
        rows = cursor.fetchall()

    conn.close()

    return [{"title": r[1], "text": r[2], "score": abs(r[3])} for r in rows]




# --- MOCK ФУНКЦИИ (Заглушки до готовности кода команды) ---
def mock_vector_search(query):
    # Имитация работы векторного поиска (Человек 5)
    # Попробуй поменять score, чтобы протестировать Fallback
    return {
        "max_score": 0.85,
        "results": [
            {"title": "Machine Learning", "score": 0.85, "text": "Machine learning is a field of study..."},
            {"title": "Artificial Intelligence", "score": 0.75, "text": "AI intelligence involves..."},
            {"title": "Neural Networks", "score": 0.60, "text": "A neural network is..."}
        ]
    }


def mock_keyword_search(query):
    # Имитация классического BM25/TF-IDF поиска
    return {
        "max_score": 1.0, # Для BM25 скор считается иначе
        "results": [
            {"title": "Keyword Match 1", "score": 5.4, "text": "Exact keyword match text..."},
            {"title": "Keyword Match 2", "score": 4.1, "text": "Another exact match..."}
        ]
    }


# --- ЛОГИКА FALLBACK ---

CONFIDENCE_THRESHOLD = 0.30


def retrieve_context(query):
    vector_data = mock_vector_search(query)

    # Узел принятия решений
    if vector_data["max_score"] < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low semantic confidence. Falling back to keyword search.", icon="⚠️")
        return mock_keyword_search(query)["results"]
    else:
        return vector_data["results"]


# --- ИНТЕГРАЦИЯ С OLLAMA ---

def generate_answer(query, context_chunks):
    # Берем только Топ-3
    top_3 = context_chunks[:3]

    # Формируем контекст для промпта
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

    # Запрос к локальной Ollama (модель gemma:4b или аналогичная)
    payload = {
        "model": "gemma",  # Укажи точное имя модели, которая установлена в Ollama
        "prompt": f"{system_prompt}\n\nUser Query: {query}",
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "Error generating response."), top_3
    except requests.exceptions.RequestException as e:
        return f"Ollama Connection Error: {e}", top_3


# --- UI ИНТЕРФЕЙС (STREAMLIT) ---

st.set_page_config(page_title="Local Wiki RAG", layout="wide")
st.title("📚 Local Wiki AI Search")

# Инициализация истории чата
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ввод пользователя
if prompt := st.chat_input("Ask me about anything in the local Wikipedia..."):
    # Добавляем запрос юзера в UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching local databases..."):
            # 1. Поиск (с логикой Fallback)
            retrieved_chunks = retrieve_context(prompt)

            # 2. Генерация ответа
            answer, used_sources = generate_answer(prompt, retrieved_chunks)

            # 3. Вывод ответа
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # 4. Визуализация источников (Accordion)
            st.markdown("### 📑 Sources Used")
            for i, source in enumerate(used_sources):
                # Создаем раскрывающуюся карточку для каждого источника
                with st.expander(f"[{i + 1}] {source['title']} (Score: {source['score']:.2f})"):
                    st.write(source['text'])
