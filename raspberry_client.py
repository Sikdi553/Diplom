import numpy as np
import sounddevice as sd
import json
import serial
import os
import sys
import warnings
import webrtcvad
import collections
import requests
import io
import scipy.io.wavfile as wavfile

warnings.filterwarnings("ignore")

# --- НАСТРОЙКИ ---
VDS_API = "http://89.124.113.21:8000"
UART_PORT = "/dev/ttyUSB0"          
BAUD_RATE = 115200                  
SAMPLING_RATE = 16000               
DURATION = 1.5                      
ENERGY_THRESHOLD = 4500 

print("====================================================")
print("     Инициализация Raspberry Cloud Robotics Client...")
print("====================================================")

# --- ИНИЦИАЛИЗАЦИЯ UART ---
try:
    ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
    print(f"✅ UART успешно подключен: {UART_PORT}")
except Exception:
    print("⚠️ ВНИМАНИЕ: UART не подключен. Работа в режиме эмуляции.")
    ser = None

vad = webrtcvad.Vad(3)

def record_audio():
    """Слушает микрофон, детектирует речь с помощью VAD и возвращает WAV байты"""
    frame_duration_ms = 30
    chunk_size = int(SAMPLING_RATE * frame_duration_ms / 1000)
    ring_buffer = collections.deque(maxlen=10) 
    
    audio_buffer = []
    triggered = False
    silence_frames = 0
    max_silence_frames = int(0.6 / (frame_duration_ms / 1000))
    max_record_frames = int(2.0 / (frame_duration_ms / 1000))
    
    print("\n[Ожидание вашей команды...]")
    stream = sd.InputStream(samplerate=SAMPLING_RATE, channels=1, dtype='int16', blocksize=chunk_size)
    stream.start()
    
    while True:
        chunk, _ = stream.read(chunk_size)
        chunk_bytes = chunk.tobytes()
        
        is_speech = vad.is_speech(chunk_bytes, SAMPLING_RATE)
        chunk_arr = np.frombuffer(chunk_bytes, dtype=np.int16)
        volume = np.max(np.abs(chunk_arr))
        
        valid_voice = is_speech and (volume > ENERGY_THRESHOLD)
        
        if not triggered:
            ring_buffer.append(chunk)
            if valid_voice:
                triggered = True
                audio_buffer.extend(ring_buffer)
                ring_buffer.clear()
                print("[!] Запись началась...")
        else:
            audio_buffer.append(chunk)
            if not valid_voice:
                silence_frames += 1
            else:
                silence_frames = 0
            
            if silence_frames > max_silence_frames or len(audio_buffer) > max_record_frames:
                break
                
    stream.stop()
    stream.close()
    
    # Сборка аудио данных
    audio_data = np.concatenate(audio_buffer).flatten()
    
    # Приведение к фиксированной длительности 1.5 с
    target_length = int(DURATION * SAMPLING_RATE)
    if len(audio_data) < target_length:
        padding = np.zeros(target_length - len(audio_data), dtype=np.int16)
        audio_data = np.concatenate((audio_data, padding))
    else:
        audio_data = audio_data[:target_length]
        
    # Записываем WAV-файл в память (BytesIO)
    wav_io = io.BytesIO()
    wavfile.write(wav_io, SAMPLING_RATE, audio_data)
    wav_io.seek(0)
    
    return wav_io.getvalue()

def main():
    print("\n🚀 Система готова. Произнесите команду в микрофон!")
    while True:
        try:
            # 1. Локальная запись голоса с помощью VAD
            wav_bytes = record_audio()
            
            # 2. Мгновенная отправка в Облако на VDS
            print("⏳ Отправка аудио на семантический анализ VDS...")
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            
            response = requests.post(f"{VDS_API}/process_audio", files=files, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "success":
                    text = result.get("text")
                    command = result.get("command")
                    char = result.get("char")
                    score = result.get("confidence", 0.0)
                    synonym = result.get("matched_synonym")
                    
                    print(f"💬 Услышано: \"{text}\"")
                    print(f"🎯 Намерение: {command} (фраза: '{synonym}', уверенность: {score*100:.1f}%)")
                    
                    # Отправка символа в UART робота
                    if char:
                        if ser:
                            ser.write(char.encode('ascii'))
                            print(f"🔌 Отправлен символ в UART: '{char}'")
                        else:
                            print(f"🔌 [Эмуляция] Отправлен бы символ в UART: '{char}'")
                            
                elif result.get("status") == "unclear":
                    text = result.get("text")
                    closest = result.get("closest_synonym")
                    score = result.get("confidence", 0.0)
                    print(f"💬 Услышано: \"{text}\"")
                    print(f"🤷‍♂️ Смысл команды не ясен. Ближайшее совпадение с '{closest}' ({score*100:.1f}%)")
                    
                else:
                    print(f"❌ Ошибка распознавания: {result.get('error')}")
            else:
                print(f"❌ Ошибка сервера: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Ошибка сети: Нет связи с VDS-сервером ({e})")
        except Exception as e:
            print(f"❌ Системная ошибка: {e}")

if __name__ == "__main__":
    main()
