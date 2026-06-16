import torch
from transformers import Wav2Vec2Model
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import warnings
warnings.filterwarnings("ignore")

# 1. Загрузка и Прунинг
print("1. Загрузка и прунинг модели (оставляем 6 слоев)...")
model_id = "facebook/wav2vec2-base"
model = Wav2Vec2Model.from_pretrained(model_id)

model.encoder.layers = model.encoder.layers[:6]
model.eval()

import sys
sys.stdout.reconfigure(encoding='utf-8')

import tempfile
import shutil

# 2. Экспорт в ONNX
print("2. Экспорт в ONNX (во временную папку для обхода ошибки кириллицы)...")
dummy_input = torch.randn(1, 24000) 
script_dir = os.path.dirname(os.path.abspath(__file__))
final_quantized_onnx = os.path.join(script_dir, "model_optimized.onnx")

# Работаем во временной папке (в пути не будет русских букв "Диплом", которые ломают ONNX)
temp_dir = tempfile.gettempdir()
raw_onnx = os.path.join(temp_dir, "model_raw.onnx")
quantized_onnx = os.path.join(temp_dir, "model_optimized.onnx")

torch.onnx.export(
    model, 
    dummy_input, 
    raw_onnx,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input_values'],
    output_names=['last_hidden_state'],
    fallback=True
)

# ХАК ДЛЯ RASPBERRY PI: Понижаем IR-версию файла до 9
print("-> Понижение IR-версии с 10 до 9 для Raspberry Pi...")
import onnx
onnx_model = onnx.load(raw_onnx)
onnx_model.ir_version = 9
onnx.save(onnx_model, raw_onnx)

# 3. Умное Квантование (Mixed Precision)
print("3. Динамическое квантование в INT8 (только MatMul-слои)...")
quantize_dynamic(
    model_input=raw_onnx,
    model_output=quantized_onnx,
    weight_type=QuantType.QUInt8,
    op_types_to_quantize=['MatMul']
)

# Перемещаем результат обратно в нашу папку
shutil.copy(quantized_onnx, final_quantized_onnx)

# Итоги
raw_size = os.path.getsize(raw_onnx) / (1024 * 1024)
quant_size = os.path.getsize(final_quantized_onnx) / (1024 * 1024)

print(f"\n======= ГОТОВО =======")
print(f"Размер исходной (6 слоев): {raw_size:.2f} MB")
print(f"Размер оптимизированной (INT8, 6 слоев): {quant_size:.2f} MB")
print(f"Файл для робота: {final_quantized_onnx}")

# Очистка
if os.path.exists(raw_onnx): os.remove(raw_onnx)
if os.path.exists(quantized_onnx): os.remove(quantized_onnx)