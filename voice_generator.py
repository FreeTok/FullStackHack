import os
import tempfile
import shutil
import subprocess
import base64
import requests
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Yandex API credentials
API_KEY = ""
FOLDER_ID = ""

# Настройки голосов для TTS
TTS_VOICES = {
    "cheb": "alena",      # Чебурашка
    "gena": "ermil",      # Крокодил Гена
    "shap": "jane",       # Шапокляк
    "volc": "filipp"      # Волк
}

# Настройки моделей для RVC
RVC_MODELS = {
    "cheb": {"model": "cheb", "has_index": True},
    "gena": {"model": "gena", "has_index": True},
    "shap": {"model": "shap", "has_index": True},
    "volc": {"model": "volc", "has_index": False}
}

def run_cmd(cmd):
    """Запуск команды"""
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("ERROR:", proc.stderr.decode('utf-8', errors='ignore'))
        raise RuntimeError("Command failed")
    return proc.stdout

def rvc_convert(input_audio, output_audio, model_name, has_index):
    """RVC конвертация через RvcWebUI localhost API"""
    try:
        # Выбор голоса
        print(f"Выбираем голос: {model_name}.pth")
        response = requests.post("http://localhost:7897/run/infer_set", json={
            "data": [f"{model_name}.pth", 0.33, 0.33]
        }, timeout=30)
        
        if response.status_code != 200 or response.json().get('data') is None:
            raise RuntimeError(f"Не удалось выбрать голос {model_name}")
        
        # Переозвучка
        index_path = f"logs/{model_name}.index" if has_index else ""
        
        response = requests.post("http://localhost:7897/run/infer_convert", json={
            "data": [
                0,              # pitch
                input_audio,    # input path
                0,
                None,
                "pm",          # f0_method
                "",
                index_path,    # index path
                0.85,          # index_rate
                3,             # filter_radius
                0,             # resample_sr
                0.25,          # rms_mix_rate
                0.33,          # protect
            ]
        }, timeout=120)
        
        if response.status_code != 200 or response.json().get('data') is None:
            raise RuntimeError("Не удалось выполнить переозвучку")
        
        # Получаем путь к результату
        result_data = response.json()["data"]
        revoiced_path = result_data[1]["name"]
        
        # Копируем результат
        shutil.copy(revoiced_path, output_audio)
        return output_audio
        
    except Exception as e:
        print(f"RVC conversion failed: {e}")
        if os.path.exists(input_audio) and input_audio != output_audio:
            shutil.copy(input_audio, output_audio)
        return output_audio

@app.route('/')
def index():
    """Главная страница с интерфейсом"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Генератор голосов персонажей</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        select, textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            box-sizing: border-box;
        }
        textarea {
            min-height: 100px;
            resize: vertical;
            font-family: Arial, sans-serif;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            display: none;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status.processing {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        audio {
            width: 100%;
            margin-top: 15px;
        }
        .download-btn {
            margin-top: 10px;
            background: #2196F3;
        }
        .download-btn:hover {
            background: #0b7dda;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Генератор голосов</h1>
        
        <div class="form-group">
            <label for="character">Выберите персонажа:</label>
            <select id="character">
                <option value="cheb">Чебурашка</option>
                <option value="gena">Крокодил Гена</option>
                <option value="shap">Шапокляк</option>
                <option value="volc">Волк</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="text">Текст для озвучки:</label>
            <textarea id="text" placeholder="Введите текст, который должен сказать персонаж..."></textarea>
        </div>
        
        <button id="generateBtn" onclick="generate()">Создать аудио</button>
        
        <div id="status" class="status"></div>
        
        <div id="audioContainer" style="display: none;">
            <audio id="audioPlayer" controls></audio>
            <button class="download-btn" onclick="download()">Скачать аудио</button>
        </div>
    </div>

    <script>
        let currentAudio = null;
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
        }
        
        function hideStatus() {
            document.getElementById('status').style.display = 'none';
        }
        
        async function generate() {
            const character = document.getElementById('character').value;
            const text = document.getElementById('text').value.trim();
            
            if (!text) {
                showStatus('Введите текст!', 'error');
                return;
            }
            
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            showStatus('Генерация аудио... Подождите...', 'processing');
            document.getElementById('audioContainer').style.display = 'none';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        character: character,
                        text: text
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    currentAudio = result.audio_base64;
                    const audioPlayer = document.getElementById('audioPlayer');
                    audioPlayer.src = 'data:audio/mp3;base64,' + result.audio_base64;
                    document.getElementById('audioContainer').style.display = 'block';
                    showStatus('✅ Аудио успешно создано!', 'success');
                } else {
                    showStatus('❌ Ошибка: ' + result.detail, 'error');
                }
            } catch (error) {
                showStatus('❌ Ошибка: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
            }
        }
        
        function download() {
            if (!currentAudio) return;
            
            const character = document.getElementById('character').value;
            const link = document.createElement('a');
            link.href = 'data:audio/mp3;base64,' + currentAudio;
            link.download = character + '_' + Date.now() + '.mp3';
            link.click();
        }
    </script>
</body>
</html>
    ''')

@app.route('/generate', methods=['POST'])
def generate():
    """Генерация аудио"""
    try:
        data = request.json
        character = data.get('character')
        text = data.get('text')
        
        if not character or not text:
            return jsonify({"detail": "Не указан персонаж или текст"}), 400
        
        if character not in TTS_VOICES:
            return jsonify({"detail": "Неизвестный персонаж"}), 400
        
        # Создаём временную директорию
        temp_dir = tempfile.mkdtemp()
        
        try:
            print(f"\n{'='*60}")
            print(f"Генерация аудио для персонажа: {character}")
            print(f"Текст: {text}")
            print(f"{'='*60}\n")
            
            # 1. Text-to-Speech через Яндекс
            print("[1/3] Синтез речи через Яндекс TTS...")
            selected_voice = TTS_VOICES[character]
            
            tts_response = requests.post(
                "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize",
                headers={"Authorization": f"Api-Key {API_KEY}"},
                data={
                    "text": text,
                    "lang": "ru-RU",
                    "voice": selected_voice,
                    "folderId": FOLDER_ID,
                    "format": "oggopus",
                    "sampleRateHertz": "48000"
                },
                timeout=30
            )
            
            if tts_response.status_code != 200:
                return jsonify({"detail": f"TTS error: {tts_response.text}"}), 500
            
            # Сохраняем TTS результат
            tts_ogg = os.path.join(temp_dir, "tts.ogg")
            with open(tts_ogg, "wb") as f:
                f.write(tts_response.content)
            
            print("✓ TTS завершён")
            
            # 2. Конвертируем OGG в WAV для RVC
            print("[2/3] Конвертация в WAV...")
            tts_wav = os.path.join(temp_dir, "tts.wav")
            run_cmd(["ffmpeg", "-y", "-i", tts_ogg, "-ar", "40000", "-ac", "1", tts_wav])
            print("✓ Конвертация завершена")
            
            # 3. Применяем RVC
            print("[3/3] Клонирование голоса через RVC...")
            final_audio = tts_wav
            model_config = RVC_MODELS.get(character)
            
            if model_config:
                rvc_out = os.path.join(temp_dir, "rvc_out.wav")
                try:
                    final_audio = rvc_convert(
                        tts_wav,
                        rvc_out,
                        model_config["model"],
                        model_config.get("has_index", False)
                    )
                    print("✓ RVC завершён")
                except Exception as e:
                    print(f"RVC failed, используем оригинальный TTS: {e}")
            
            # 4. Конвертируем в MP3
            final_mp3 = os.path.join(temp_dir, "final.mp3")
            run_cmd(["ffmpeg", "-y", "-i", final_audio, "-acodec", "libmp3lame", "-b:a", "128k", final_mp3])
            
            # 5. Читаем и кодируем в base64
            with open(final_mp3, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            print(f"\n{'='*60}")
            print("🎯 ГОТОВО!")
            print(f"{'='*60}\n")
            
            return jsonify({"audio_base64": audio_b64})
            
        finally:
            # Удаляем временные файлы
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"detail": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎙️  ГЕНЕРАТОР ГОЛОСОВ ПЕРСОНАЖЕЙ")
    print("="*60)
    print("\n📌 Откройте в браузере: http://localhost:5000")
    print("\n⚠️  Убедитесь, что RvcWebUI запущен на порту 7897!")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
