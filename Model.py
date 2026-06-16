import sounddevice as sd
import numpy as np
import collections
import webrtcvad
import warnings
import json
import os
import sys
import scipy.io.wavfile as wavfile

warnings.filterwarnings("ignore")

# --- НАСТРОЙКИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")         

SAMPLING_RATE = 16000               
DURATION = 1.5                      

os.makedirs(DATASET_DIR, exist_ok=True)

class LocalDatasetRecorder:
    def __init__(self):
        self.vad = webrtcvad.Vad(3) 

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "commands" not in data:
                        return {"commands": {}, "classes_list": []}
                    return data
            except:
                pass
        return {"commands": {}, "classes_list": []}

    def save_config(self, cfg):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def record_audio(self):
        """Умная запись: VAD + Шумоподавитель"""
        frame_duration_ms = 30
        chunk_size = int(SAMPLING_RATE * frame_duration_ms / 1000)
        ring_buffer = collections.deque(maxlen=10) 
        
        audio_buffer = []
        triggered = False
        silence_frames = 0
        
        max_silence_frames = int(0.6 / (frame_duration_ms / 1000))
        max_record_frames = int(2.0 / (frame_duration_ms / 1000))
        
        ENERGY_THRESHOLD = 4500 
        
        print("   [Ожидание голоса...]")
        
        stream = sd.InputStream(samplerate=SAMPLING_RATE, channels=1, dtype='int16', blocksize=chunk_size)
        stream.start()
        
        while True:
            chunk, _ = stream.read(chunk_size)
            chunk_bytes = chunk.tobytes()
            
            is_speech = self.vad.is_speech(chunk_bytes, SAMPLING_RATE)
            chunk_arr = np.frombuffer(chunk_bytes, dtype=np.int16)
            volume = np.max(np.abs(chunk_arr))
            
            valid_voice = is_speech and (volume > ENERGY_THRESHOLD)
            
            if not triggered:
                ring_buffer.append(chunk)
                if valid_voice:
                    triggered = True
                    audio_buffer.extend(ring_buffer)
                    ring_buffer.clear()
                    print("   [Запись...]")
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
        
        audio_data = np.concatenate(audio_buffer).flatten()
        return audio_data

    def create_command(self):
        cfg = self.load_config()
        print("\n=== СОЗДАНИЕ КОМАНДЫ ===")
        cmd_name = input("Введите название (например, 'вперед'): ").strip().lower()
        if not cmd_name: return
        
        char = input(f"Какую английскую букву для '{cmd_name}'? ").strip().lower()
        if len(char) != 1:
            print("Ошибка: нужна ровно 1 буква!")
            return
            
        cfg["commands"][cmd_name] = char
        self.save_config(cfg)
        
        os.makedirs(os.path.join(DATASET_DIR, cmd_name), exist_ok=True)
        print(f"Команда '{cmd_name}' -> '{char}' создана.")

    def train_command(self):
        cfg = self.load_config()
        cmds = list(cfg["commands"].keys())
        if not cmds:
            print("База пуста! Создайте команду.")
            return
            
        print("\n=== ДОБАВЛЕНИЕ ГОЛОСА ===")
        for i, cmd in enumerate(cmds):
            count = 0
            d = os.path.join(DATASET_DIR, cmd)
            if os.path.exists(d): count = len(os.listdir(d))
            print(f"{i+1}. {cmd} (Записей: {count})")
            
        try:
            idx = int(input("Выберите номер: ")) - 1
            if 0 <= idx < len(cmds):
                target_cmd = cmds[idx]
                target_dir = os.path.join(DATASET_DIR, target_cmd)
                os.makedirs(target_dir, exist_ok=True)
                
                input("Нажмите Enter и скажите...")
                audio = self.record_audio()
                
                count = len(os.listdir(target_dir)) + 1
                fpath = os.path.join(target_dir, f"{count}.wav")
                
                wavfile.write(fpath, SAMPLING_RATE, audio)
                print(f"Запись #{count} сохранена!")
            else:
                print("Неверный номер.")
        except ValueError:
            print("Ошибка.")

if __name__ == "__main__":
    app = LocalDatasetRecorder()
    while True:
        print("\n======= СБОР ДАННЫХ =======")
        print("1. Создать команду (назначить букву UART)")
        print("2. Записать голос в датасет")
        print("0. ВЫХОД")
        cmd = input("> ")
        if cmd == '1': app.create_command()
        elif cmd == '2': app.train_command()
        elif cmd == '0': break
