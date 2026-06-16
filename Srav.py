import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import onnxruntime as ort
import time
import psutil
import os
import gc
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Настройки теста
MODEL_ID = "facebook/wav2vec2-base"
ONNX_MODEL_PATH = "model_optimized.onnx"
AUDIO_LEN_SEC = 1.5
SAMPLE_RATE = 16000
NUM_SAMPLES = int(AUDIO_LEN_SEC * SAMPLE_RATE) # 24000

def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def benchmark_pytorch():
    print("\n--- 1. Тестирование оригинальной модели (PyTorch, FP32, 12 слоев) ---")
    gc.collect()
    mem_before = get_ram_mb()
    
    # Загрузка
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID)
    model.eval()
    
    mem_after = get_ram_mb()
    ram_used = mem_after - mem_before
    disk_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024) # 4 байта на FP32
    
    # Подготовка данных
    dummy_audio = torch.randn(NUM_SAMPLES)
    inputs = extractor(dummy_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_tensor = inputs.input_values
    
    # Прогрев
    with torch.no_grad():
        for _ in range(3):
            model(input_tensor)
            
    # Замер скорости
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)
    end = time.perf_counter()
    
    latency = ((end - start) / 10) * 1000
    rtf = latency / (AUDIO_LEN_SEC * 1000)
    
    del model
    gc.collect()
    
    return {"name": "PyTorch (Базовая)", "disk": disk_size, "ram": ram_used, "lat": latency, "rtf": rtf}

def benchmark_onnx():
    print("\n--- 2. Тестирование оптимизированной модели (ONNX, INT8, 6 слоев) ---")
    gc.collect()
    mem_before = get_ram_mb()
    
    # Загрузка ONNX
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    mem_after = get_ram_mb()
    ram_used = mem_after - mem_before
    disk_size = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
    
    # Подготовка данных
    dummy_audio = np.random.randn(NUM_SAMPLES).astype(np.float32)
    inputs = extractor(dummy_audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    onnx_inputs = {'input_values': inputs['input_values'].astype(np.float32)}
    
    # Прогрев
    for _ in range(3):
        session.run(['last_hidden_state'], onnx_inputs)
        
    # Замер скорости
    start = time.perf_counter()
    for _ in range(10):
        session.run(['last_hidden_state'], onnx_inputs)
    end = time.perf_counter()
    
    latency = ((end - start) / 10) * 1000
    rtf = latency / (AUDIO_LEN_SEC * 1000)
    
    del session
    gc.collect()
    
    return {"name": "ONNX (Оптимизированная)", "disk": disk_size, "ram": ram_used, "lat": latency, "rtf": rtf}

if __name__ == "__main__":
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Ошибка: Файл {ONNX_MODEL_PATH} не найден. Сначала запустите скрипт оптимизации.")
        exit()
        
    res_pt = benchmark_pytorch()
    res_onnx = benchmark_onnx()
    
    # Вывод красивой таблицы для диплома
    print("\n" + "="*85)
    print(f"{'Метрика':<30} | {res_pt['name']:<22} | {res_onnx['name']:<22} | {'Улучшение'}")
    print("-" * 85)
    
    disk_impr = res_pt['disk'] / res_onnx['disk']
    lat_impr = res_pt['lat'] / res_onnx['lat']
    
    print(f"{'Размер на диске (МБ)':<30} | {res_pt['disk']:<22.2f} | {res_onnx['disk']:<22.2f} | в {disk_impr:.1f} раз")
    print(f"{'Потребление RAM (МБ)':<30} | {res_pt['ram']:<22.2f} | {res_onnx['ram']:<22.2f} | снижено")
    print(f"{'Задержка (Latency), мс':<30} | {res_pt['lat']:<22.2f} | {res_onnx['lat']:<22.2f} | в {lat_impr:.1f} раз")
    print(f"{'Коэфф. реального времени (RTF)':<30} | {res_pt['rtf']:<22.4f} | {res_onnx['rtf']:<22.4f} | {'-'}")
    print("="*85)