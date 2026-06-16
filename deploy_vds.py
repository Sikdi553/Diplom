import os
import sys
import time

try:
    import paramiko
    from scp import SCPClient
except ImportError:
    print("Библиотеки paramiko или scp не найдены!")
    print("Пожалуйста, установите их командой: pip install paramiko scp")
    sys.exit(1)

# Принудительно настраиваем UTF-8 для вывода в консоль Windows, если это возможно
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Данные подключения к VDS
VDS_IP = "89.124.113.21"
VDS_USER = "root"
VDS_PASS = "7__2k37pSB3+6uYizkMu"

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_DIR = "/root/diplom"

files_to_upload = [
    ("bot_vk.py", "bot_vk.py"),
    ("config.json", "config.json")
]

def run_ssh_command(ssh_client, command):
    print(f"Executing: {command}")
    stdin, stdout, stderr = ssh_client.exec_command(command)
    
    # Читаем вывод в реальном времени
    while True:
        line = stdout.readline()
        if not line:
            break
        print(f"  [OUT] {line.strip()}")
        
    err = stderr.read().decode('utf-8', errors='ignore')
    if err:
        print(f"  [ERR] {err.strip()}")

def main():
    print("=== AUTOMATIC DEPLOY TO VDS ===")
    print(f"Connecting to {VDS_IP} as {VDS_USER}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(VDS_IP, username=VDS_USER, password=VDS_PASS, timeout=15)
        print("[OK] SSH connection established successfully!")
    except Exception as e:
        print(f"[ERROR] SSH connection failed: {e}")
        sys.exit(1)

    # 1. Обновление пакетов и установка системных зависимостей
    print("\n1. Updating apt packages and installing ffmpeg, pip...")
    run_ssh_command(ssh, "apt-get update && apt-get install -y python3-pip python3-venv ffmpeg")

    # 2. Создание рабочей директории
    print("\n2. Creating working directory on server...")
    run_ssh_command(ssh, f"mkdir -p {REMOTE_DIR}")

    # 3. Инициализация виртуального окружения
    print("\n3. Creating Python virtual environment on VDS...")
    run_ssh_command(ssh, f"python3 -m venv {REMOTE_DIR}/venv")

    # 4. Установка Python-библиотек
    # Ставим CPU-версию PyTorch для экономии места и времени, затем transformers, onnxruntime, scikit-learn и skl2onnx
    print("\n4. Installing AI libraries (this can take 1-2 minutes)...")
    pip_path = f"{REMOTE_DIR}/venv/bin/pip"
    run_ssh_command(ssh, f"{pip_path} install --upgrade pip")
    run_ssh_command(ssh, f"{pip_path} install torch --index-url https://download.pytorch.org/whl/cpu")
    run_ssh_command(ssh, f"{pip_path} install transformers vk_api pydub fastapi uvicorn scikit-learn pyserial faster-whisper python-multipart")

    # 5. Копирование локальных файлов на VDS
    print("\n5. Copying project files to VDS via SFTP...")
    try:
        with SCPClient(ssh.get_transport()) as scp:
            for local_file, remote_file in files_to_upload:
                local_path = os.path.join(LOCAL_DIR, local_file)
                # ХАК: Для VDS (Linux) путь ВСЕГДА должен формироваться через прямой слэш '/', а не через os.path.join
                remote_path = REMOTE_DIR + "/" + remote_file
                
                if os.path.exists(local_path):
                    print(f"  Uploading {local_file} -> {remote_path} ({os.path.getsize(local_path)/(1024*1024):.2f} MB)...")
                    scp.put(local_path, remote_path)
                else:
                    if local_file == "config.json":
                        print("  config.json not found locally. Creating default config on VDS...")
                        run_ssh_command(ssh, f"echo '{{\"commands\": {{}}, \"classes_list\": []}}' > {remote_path}")
                    else:
                        print(f"  [ERROR] Critical local file {local_file} not found!")
                        sys.exit(1)
        print("[OK] All files successfully uploaded!")
    except Exception as e:
        print(f"[ERROR] Failed to copy files: {e}")
        sys.exit(1)

    # 6. Настройка и запуск службы systemd
    print("\n6. Creating systemd service for VK bot...")
    service_content = """[Unit]
Description=VK Bot and Training Agent for Wav2Vec2
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/diplom
ExecStart=/root/diplom/venv/bin/python /root/diplom/bot_vk.py
Restart=always

[Install]
WantedBy=multi-user.target
"""
    # Записываем службу во временный файл и перемещаем
    run_ssh_command(ssh, f"echo '{service_content}' > /tmp/vkbot.service")
    run_ssh_command(ssh, "mv /tmp/vkbot.service /etc/systemd/system/vkbot.service")
    
    print("  Reloading systemctl daemon...")
    run_ssh_command(ssh, "systemctl daemon-reload")
    print("  Enabling vkbot.service...")
    run_ssh_command(ssh, "systemctl enable vkbot.service")
    print("  Starting vkbot.service...")
    run_ssh_command(ssh, "systemctl restart vkbot.service")

    # 7. Проверка статуса службы
    print("\n7. Checking service status...")
    time.sleep(3) # Даем боту время запуститься
    run_ssh_command(ssh, "systemctl status vkbot.service")

    ssh.close()
    print("\n=== DEPLOY COMPLETED SUCCESSFULLY ===")
    print("VK Bot and Training Agent are running on VDS 24/7!")
    print("Model Distribution API: http://89.124.113.21:8000/status")

if __name__ == "__main__":
    main()
