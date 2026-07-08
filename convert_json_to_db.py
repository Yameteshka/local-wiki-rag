import sqlite3
import json
import os
import re

JSON_PATH = "data/wikipedia_metadata_manifest.json"
DB_PATH = "wikipedia.db"
BATCH_SIZE = 10000  # Фиксируем размер пачки, чтобы не забивать RAM

if not os.path.exists(JSON_PATH):
    print(f"❌ Файл {JSON_PATH} не найден в корне!")
    exit()

print("📦 Инициализация базы данных SQLite...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS articles;")
cursor.execute("""
    CREATE VIRTUAL TABLE articles USING fts5(
        id UNINDEXED, 
        title, 
        text
    );
""")


def stream_json_records(filepath):
    """
    Потоковый парсер: читает файл блоками, отслеживает баланс скобок
    и извлекает статьи по одной, не загружая файл целиком в память.
    """
    # 1. Определяем структуру: глобальный массив [..] или словарь {..}
    with open(filepath, 'r', encoding='utf-8') as f:
        file_type = None
        while True:
            char = f.read(1)
            if not char: break
            if char.strip():
                if char == '[':
                    file_type = 'array'
                elif char == '{':
                    file_type = 'dict'
                break

    if file_type is None:
        return

    print(f"🔍 Структура манифеста: {file_type.upper()}. Запуск потокового чтения...")

    # 2. Читаем и парсим блоки данных
    with open(filepath, 'r', encoding='utf-8') as f:
        brace_count = 0
        in_string = False
        escaped = False
        obj_chars = []
        prefix_chars = []

        target_depth = 1 if file_type == 'array' else 2
        max_prefix_len = 200

        while True:
            chunk = f.read(256 * 1024)  # Читаем по 256 КБ
            if not chunk:
                break

            for char in chunk:
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == '\\':
                        escaped = True
                    elif char == '"':
                        in_string = False

                    if brace_count >= target_depth:
                        obj_chars.append(char)
                    else:
                        prefix_chars.append(char)
                        if len(prefix_chars) > max_prefix_len: prefix_chars.pop(0)
                    continue

                if char == '"':
                    in_string = True
                    if brace_count >= target_depth:
                        obj_chars.append(char)
                    else:
                        prefix_chars.append(char)
                        if len(prefix_chars) > max_prefix_len: prefix_chars.pop(0)
                elif char == '{':
                    brace_count += 1
                    if brace_count >= target_depth:
                        obj_chars.append(char)
                elif char == '}':
                    if brace_count >= target_depth:
                        obj_chars.append(char)

                    brace_count -= 1

                    if brace_count == (target_depth - 1):
                        obj_str = "".join(obj_chars)
                        obj_chars = []

                        try:
                            record = json.loads(obj_str)

                            # Извлекаем ID (из ключа для dict или из поля для array)
                            if file_type == 'dict':
                                prefix_str = "".join(prefix_chars)
                                keys = re.findall(r'"([^"]+)"\s*:', prefix_str)
                                rec_id = keys[-1] if keys else None
                            else:
                                rec_id = record.get('id')

                            if rec_id is None and 'id' in record:
                                rec_id = record['id']

                            yield rec_id, record.get('title', ''), record.get('text', '')
                        except:
                            pass  # Игнорируем битые объекты, если они есть

                        prefix_chars = []
                else:
                    if brace_count >= target_depth:
                        obj_chars.append(char)
                    else:
                        prefix_chars.append(char)
                        if len(prefix_chars) > max_prefix_len: prefix_chars.pop(0)


# 3. Собираем записи пачками и пишем в базу
batch = []
count = 0

for rec_id, title, text in stream_json_records(JSON_PATH):
    batch.append((rec_id, title, text))
    count += 1

    if len(batch) >= BATCH_SIZE:
        cursor.executemany("INSERT INTO articles (id, title, text) VALUES (?, ?, ?);", batch)
        conn.commit()
        print(f"⚡ Залито статей в базу: {count}...")
        batch = []

# Дозаписываем остатки
if batch:
    cursor.executemany("INSERT INTO articles (id, title, text) VALUES (?, ?, ?);", batch)
    conn.commit()

conn.close()
print(f"✅ Успех! База {DB_PATH} успешно создана. Всего обработано статей: {count}")