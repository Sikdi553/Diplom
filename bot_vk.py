import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id
import requests
import numpy as np
import json
import serial
import os
import sys
import io
import sqlite3
import threading
from pydub import AudioSegment
import warnings
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Импорты ИИ моделей для ASR и NLU
from faster_whisper import WhisperModel
import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# --- НАСТРОЙКИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")         
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DB_PATH = os.path.join(BASE_DIR, "database.db")

UART_PORT = "/dev/ttyUSB0"          
BAUD_RATE = 115200                  
SAMPLING_RATE = 16000               

# --- ТОКЕНЫ ВК ---
VK_TOKEN = "vk1.a.Yd4pC47JMeC9H0IFZr9kYGR8eGxgfh52UsgSm7BetGD2j3z5EoMemxo3NWk-21YjNqto74lv8xIKFt9TfBvM8eeEZ0nKdCTggnGUX9hBRilKvTKJn7Rv6bszrzb1h6ydKnozE3xkomRd-06F7nniMzc0qWtL57WV8Q174_jx0SMoe5lfKid2kkwuzgLP5IgDzq-9kIPLSDvOYCm9cZHH9A" 
GROUP_ID = "237283505" 

os.makedirs(DATASET_DIR, exist_ok=True)

# --- ИНИЦИАЛИЗАЦИЯ ИИ МОДЕЛЕЙ НА СЕРВЕРЕ ---
print("Загрузка Speech-to-Text модели Whisper (tiny)...")
# tiny версия весит всего 70 МБ и идеально работает на CPU в реальном времени
whisper_model = WhisperModel("tiny", device="cpu", compute_type="float32")

print("Загрузка NLU модели понимания русского языка RuBERT-tiny2...")
# Легкая модель для извлечения смысла предложений (110 МБ)
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
rubert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

# --- ИНИЦИАЛИЗАЦИЯ UART (с эмуляцией) ---
try:
    ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
    print(f"UART подключен: {UART_PORT}")
except Exception:
    print("ВНИМАНИЕ: UART не подключен. Работа в режиме эмуляции на VDS.")
    ser = None

# --- ИНИЦИАЛИЗАЦИЯ БД SQLITE ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS commands (
            name TEXT PRIMARY KEY,
            char TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS synonyms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command TEXT,
            text TEXT UNIQUE,
            FOREIGN KEY(command) REFERENCES commands(name) ON DELETE CASCADE
        )
    """)
    conn.commit()
    
    # Наполнение дефолтными командами и синонимами, если база синонимов пуста
    cursor.execute("SELECT COUNT(*) FROM synonyms")
    if cursor.fetchone()[0] == 0:
        print("База синонимов пуста. Наполнение дефолтными командами и семантическими синонимами...")
        default_commands = [
            ("вперед", "f"),
            ("назад", "b"),
            ("влево", "l"),
            ("вправо", "r"),
            ("стоп", "s")
        ]
        # Сохраняем пользовательские привязки (w, s, d) и добавляем отсутствующие
        cursor.executemany("INSERT OR IGNORE INTO commands (name, char) VALUES (?, ?)", default_commands)
        
        default_synonyms = [
            ("вперед", "вперед"), ("вперед", "прямо"), ("вперед", "езжай"), ("вперед", "газуй"), 
            ("вперед", "катись вперед"), ("вперед", "поехали вперед"), ("вперед", "двигайся вперед"),
            ("назад", "назад"), ("назад", "сдай назад"), ("назад", "отъезжай"), ("назад", "катись назад"), 
            ("назад", "назад сдавай"), ("назад", "назад отъезжай"),
            ("влево", "влево"), ("влево", "налево"), ("влево", "поверни влево"), ("влево", "левее"), 
            ("влево", "поворачивай влево"), ("влево", "в левую сторону"),
            ("вправо", "вправо"), ("вправо", "направо"), ("вправо", "поверни вправо"), ("вправо", "правее"), 
            ("вправо", "поворачивай вправо"), ("вправо", "в правую сторону"),
            ("стоп", "стоп"), ("стоп", "остановись"), ("стоп", "стоять"), ("стоп", "тормози"), 
            ("стоп", "замри"), ("стоп", "остановка"), ("стоп", "хватит"), ("стоп", "прекрати")
        ]
        cursor.executemany("INSERT OR IGNORE INTO synonyms (command, text) VALUES (?, ?)", default_synonyms)
        conn.commit()
        
    conn.close()

init_db()

# --- ФУНКЦИИ КОНФИГА ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception:
            pass
    return {"commands": {}, "classes_list": []}

def save_config(cfg):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def sync_config_with_db():
    cfg = load_config()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, char FROM commands")
    rows = cursor.fetchall()
    conn.close()
    
    cfg["commands"] = {name: char for name, char in rows}
    save_config(cfg)

sync_config_with_db()

# --- МАТЕМАТИКА NLU (СЕМАНТИЧЕСКИЙ АНАЛИЗ) ---
def get_sentence_embedding(text):
    """Превращает текст в 312-мерный вектор семантического смысла (RuBERT-tiny2)"""
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = rubert_model(**inputs)
    # Метод Mean Pooling для получения единого вектора предложения
    token_embeddings = outputs[0]
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().numpy()

def cosine_similarity(a, b):
    """Вычисляет косинусное сходство между двумя векторами"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

# Кэш семантических векторов синонимов для моментального инференса
synonyms_cache = {}

def rebuild_synonyms_cache():
    """Вычисляет семантические векторы для всех синонимов в БД и кэширует их"""
    global synonyms_cache
    print("Пересчет семантических векторов для базы синонимов...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT command, text FROM synonyms")
    rows = cursor.fetchall()
    conn.close()
    
    new_cache = {}
    for command, text in rows:
        try:
            vector = get_sentence_embedding(text)
            new_cache[text] = {
                "command": command,
                "vector": vector
            }
        except Exception as e:
            print(f"Ошибка кэширования синонима '{text}': {e}")
            
    synonyms_cache = new_cache
    print(f"Кэшировано {len(synonyms_cache)} синонимов.")

# Первичный расчет кэша при старте
rebuild_synonyms_cache()

def classify_intent(text, threshold=0.80):
    """Определяет наиболее близкую команду по смыслу текста"""
    if not synonyms_cache:
        return None, 0.0, "База синонимов пуста"
        
    # Приводим к нижнему регистру и очищаем знаки препинания
    clean_text = re.sub(r'[^\w\s]', '', text.lower().strip())
    if not clean_text:
        return None, 0.0, "Пустой запрос"
        
    # Вычисляем вектор входящей фразы
    vec_input = get_sentence_embedding(clean_text)
    
    best_match = None
    best_score = -1.0
    best_synonym = ""
    
    # Сравниваем со всеми закэшированными синонимами
    for syn_text, syn_data in synonyms_cache.items():
        score = cosine_similarity(vec_input, syn_data["vector"])
        if score > best_score:
            best_score = score
            best_match = syn_data["command"]
            best_synonym = syn_text
            
    print(f"[NLU] Вход: '{text}' -> Наиболее близко к синониму '{best_synonym}' (Команда: {best_match.upper()}, Сходство: {best_score:.3f})")
    
    if best_score >= threshold:
        return best_match, best_score, best_synonym
    return None, best_score, best_synonym

# --- ОБРАБОТКА ГОЛОСА ---
def transcribe_audio_bytes(audio_bytes):
    """Декодирует OGG/WAV в памяти и распознает речь с помощью Whisper"""
    # Сохраняем в памяти
    audio_file = io.BytesIO(audio_bytes)
    audio = AudioSegment.from_file(audio_file)
    
    # Сохраняем во временный wav-файл в памяти/tmp для Whisper
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    
    # Запускаем транскрипцию Whisper
    segments, info = whisper_model.transcribe(wav_io, beam_size=3, language="ru")
    text = " ".join([segment.text for segment in segments])
    return text.strip()

def send_to_atmega(char):
    if char and ser:
        ser.write(char.encode('ascii'))
        print(f"[UART] Отправлен символ: {char}")
    else:
        print(f"[UART Эмуляция] Отправлен бы символ: {char}")

# --- FASTAPI СЕРВЕР (CLOUD ROBOTICS API) ---
app_api = FastAPI(title="Wav2Vec2 Cloud NLU API")

@app_api.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """API для Raspberry Pi: Принимает записанное аудио, распознает речь и возвращает команду"""
    try:
        audio_bytes = await file.read()
        
        # 1. Speech-to-Text (Распознавание речи)
        text = transcribe_audio_bytes(audio_bytes)
        print(f"[API Cloud Robotics] Услышано: '{text}'")
        
        if not text:
            return JSONResponse(status_code=200, content={"status": "error", "error": "Тишина или речь не распознана"})
            
        # 2. Natural Language Understanding (Извлечение смысла)
        command, score, synonym = classify_intent(text)
        
        if command:
            cfg = load_config()
            char = cfg["commands"].get(command, '')
            send_to_atmega(char) # Дублируем отправку в UART на VDS, если нужно
            
            return {
                "status": "success",
                "text": text,
                "command": command.upper(),
                "char": char,
                "confidence": float(score),
                "matched_synonym": synonym
            }
        else:
            return {
                "status": "unclear",
                "text": text,
                "error": "Смысл команды не ясен",
                "confidence": float(score),
                "closest_synonym": synonym
            }
            
    except Exception as e:
        print(f"[API Error] {e}")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})

@app_api.get("/status")
def get_status():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT command, COUNT(*) FROM synonyms GROUP BY command")
    stats = cursor.fetchall()
    conn.close()
    
    return {
        "status": "online",
        "engine": "ASR (Whisper) + NLU (RuBERT-tiny2)",
        "commands": load_config().get("commands", {}),
        "synonyms_stats": {cmd: count for cmd, count in stats},
        "total_cached_synonyms": len(synonyms_cache)
    }

def run_fastapi():
    uvicorn.run(app_api, host="0.0.0.0", port=8000)

# Запуск API в фоновом потоке
api_thread = threading.Thread(target=run_fastapi, daemon=True)
api_thread.start()

# --- ИНТЕРФЕЙС ВК БОТА ---
vk_session = vk_api.VkApi(token=VK_TOKEN)
longpoll = VkBotLongPoll(vk_session, GROUP_ID)
vk = vk_session.get_api()

user_states = {}

def get_keyboard():
    keyboard = VkKeyboard(one_time=False)
    
    keyboard.add_button("Статус Агента", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("Список синонимов", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("Добавить синоним", color=VkKeyboardColor.POSITIVE)
    keyboard.add_button("Создать команду", color=VkKeyboardColor.POSITIVE)
    return keyboard.get_keyboard()

def send_msg(peer_id, message, keyboard=None):
    vk.messages.send(
        peer_id=peer_id,
        message=message,
        random_id=get_random_id(),
        keyboard=keyboard
    )

print("Бот ВК запущен и ждет сообщений на VDS в режиме семантического понимания контекста!")

# Цикл обработки ВК сообщений
for event in longpoll.listen():
    if event.type == VkBotEventType.MESSAGE_NEW:
        msg = event.obj.message
        peer_id = msg['peer_id']
        raw_text = msg.get('text', '')
        
        text = re.sub(r'^\[(?:club|public)\d+\|.*?\][,\s]*', '', raw_text)
        text = re.sub(r'^@(?:club|public)\d+[,\s]*', '', text)
        text = text.strip()
        
        print(f"[DEBUG] {peer_id}: {text}")
        state = user_states.get(peer_id, "idle")

        if text.lower() in ["начать", "меню", "привет"]:
            user_states[peer_id] = "idle"
            send_msg(peer_id, "🤖 Интеллектуальный Голосовой Ассистент на VDS.\n\n"
                              "Я переведен на режим ASR + NLU (Whisper + RuBERT)!\n"
                              "Я умею понимать контекст, синонимы и любые лишние слова. "
                              "Просто отправьте мне голосовое сообщение, сказав команду своими словами (например, 'катись вперед пожалуйста').", get_keyboard())
        
        elif text == "Создать команду":
            user_states[peer_id] = "waiting_new_cmd"
            send_msg(peer_id, "Введите название для новой базовой команды (например 'свет'):")
            
        elif state == "waiting_new_cmd":
            cmd_name = text.lower()
            user_states['temp_cmd'] = cmd_name
            user_states[peer_id] = "waiting_new_char"
            send_msg(peer_id, f"Укажите ОДНУ английскую букву, которая будет отправляться по UART при команде '{cmd_name}':")
            
        elif state == "waiting_new_char":
            char = text.lower()
            if len(char) != 1:
                send_msg(peer_id, "❌ Пожалуйста, введите ровно одну английскую букву.")
            else:
                cmd_name = user_states.get('temp_cmd', 'unknown')
                
                # Записываем в БД
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("INSERT OR REPLACE INTO commands (name, char) VALUES (?, ?)", (cmd_name, char))
                # Также сразу добавляем саму команду в список синонимов
                cursor.execute("INSERT OR IGNORE INTO synonyms (command, text) VALUES (?, ?)", (cmd_name, cmd_name))
                conn.commit()
                conn.close()
                
                # Синхронизируем конфиг и пересчитываем кэш
                sync_config_with_db()
                rebuild_synonyms_cache()
                
                user_states[peer_id] = "idle"
                send_msg(peer_id, f"✅ Базовая команда '{cmd_name}' привязана к символу '{char}' и добавлена в семантическую базу.", get_keyboard())

        elif text == "Добавить синоним":
            user_states[peer_id] = "waiting_synonym_cmd"
            cfg = load_config()
            cmds = list(cfg["commands"].keys())
            send_msg(peer_id, f"Для какой команды вы хотите добавить синоним? Доступные команды:\n{', '.join(cmds)}")
            
        elif state == "waiting_synonym_cmd":
            target_cmd = text.lower()
            cfg = load_config()
            if target_cmd not in cfg["commands"]:
                send_msg(peer_id, f"❌ Ошибка: Команда '{target_cmd}' не зарегистрирована. Отмена операции.", get_keyboard())
                user_states[peer_id] = "idle"
            else:
                user_states['temp_synonym_cmd'] = target_cmd
                user_states[peer_id] = "waiting_synonym_text"
                send_msg(peer_id, f"Введите текстовый синоним (слово или целую фразу) для команды '{target_cmd}':")
                
        elif state == "waiting_synonym_text":
            synonym_text = text.lower().strip()
            target_cmd = user_states.get('temp_synonym_cmd', 'unknown')
            
            if not synonym_text:
                send_msg(peer_id, "❌ Текст синонима не может быть пустым. Отмена.", get_keyboard())
            else:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO synonyms (command, text) VALUES (?, ?)", (target_cmd, synonym_text))
                    conn.commit()
                    conn.close()
                    
                    # Обновляем семантический кэш
                    rebuild_synonyms_cache()
                    
                    send_msg(peer_id, f"✅ Синоним '{synonym_text}' успешно привязан к команде '{target_cmd}'!", get_keyboard())
                except sqlite3.IntegrityError:
                    send_msg(peer_id, f"⚠️ Синоним '{synonym_text}' уже зарегистрирован в базе данных.", get_keyboard())
            
            user_states[peer_id] = "idle"

        elif text == "Список синонимов":
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT command, text FROM synonyms ORDER BY command")
            rows = cursor.fetchall()
            conn.close()
            
            from collections import defaultdict
            cmds_dict = defaultdict(list)
            for command, syn_text in rows:
                cmds_dict[command].append(syn_text)
                
            msg_lines = ["📋 Список зарегистрированных синонимов и фраз:"]
            for cmd, syns in cmds_dict.items():
                msg_lines.append(f"\n⚙️ {cmd.upper()}:")
                for s in syns:
                    msg_lines.append(f"  - \"{s}\"")
                    
            send_msg(peer_id, "\n".join(msg_lines), get_keyboard())

        elif text == "Статус Агента":
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT command, COUNT(*) FROM synonyms GROUP BY command")
            stats = cursor.fetchall()
            conn.close()
            
            stats_str = "\n".join([f"- {cmd}: {count} фраз-синонимов" for cmd, count in stats])
            if not stats_str:
                stats_str = "База пуста."
                
            status_msg = (
                f"📊 Интеллектуальный Агент NLU на VDS:\n\n"
                f"🧠 Модели: Whisper-tiny (ASR) + RuBERT-tiny2 (NLU)\n"
                f"📂 Статистика семантической базы:\n{stats_str}\n\n"
                f"⚡️ Всего синонимов в кэше: {len(synonyms_cache)}\n"
                f"🔌 Cloud Robotics API для робота: http://89.124.113.21:8000/status"
            )
            send_msg(peer_id, status_msg, get_keyboard())

        elif msg.get('attachments') and msg['attachments'][0]['type'] == 'audio_message':
            audio_url = msg['attachments'][0]['audio_message']['link_ogg']
            audio_bytes = requests.get(audio_url).content
            
            send_msg(peer_id, "⏳ Семантический Агент NLU обрабатывает голосовое сообщение...")
            
            try:
                # 1. Распознавание речи через Whisper
                transcribed_text = transcribe_audio_bytes(audio_bytes)
                print(f"[ASR ВК] Услышано: '{transcribed_text}'")
                
                if not transcribed_text:
                    send_msg(peer_id, "❌ Речь не распознана или вы отправили тишину. Попробуйте еще раз.", get_keyboard())
                    continue
                    
                # 2. Семантический классификатор RuBERT
                command, score, synonym = classify_intent(transcribed_text)
                
                if command:
                    cfg = load_config()
                    char = cfg["commands"].get(command, '')
                    
                    send_to_atmega(char)
                    
                    success_msg = (
                        f"💬 Вы сказали: \"{transcribed_text}\"\n\n"
                        f"🎯 Распознано намерение: {command.upper()}\n"
                        f"🔗 Совпадение с синонимом: \"{synonym}\" ({score*100:.1f}%)\n"
                        f"🔌 UART символ: '{char}'"
                    )
                    send_msg(peer_id, success_msg, get_keyboard())
                else:
                    closest_msg = ""
                    if synonym:
                        closest_msg = f"\n(Ближайший синоним: \"{synonym}\", совпадение: {score*100:.1f}%)"
                        
                    fail_msg = (
                        f"💬 Вы сказали: \"{transcribed_text}\"\n\n"
                        f"🤷‍♂️ Смысл команды не ясен. {closest_msg}\n\n"
                        f"💡 Подсказка: Вы можете нажать кнопку 'Добавить синоним' и привязать фразу \"{transcribed_text}\" к любой базовой команде!"
                    )
                    send_msg(peer_id, fail_msg, get_keyboard())
                    
            except Exception as e:
                print(f"[VK Error] {e}")
                send_msg(peer_id, f"❌ Системная ошибка Агента: {str(e)}", get_keyboard())