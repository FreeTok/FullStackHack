import io
import os
import sys
import subprocess
import base64
import tempfile
import httpx
import requests
import torch
import soundfile as sf
import time
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydub import AudioSegment
import uvicorn
import fairseq
import random
import shutil

torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])

app = FastAPI(title="Cheburashka AI with Nano Banana")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Yandex API credentials
API_KEY = ""
FOLDER_ID = ""

# OpenRouter API credentials
OPENROUTER_API_KEY = ""  # Получить на https://openrouter.ai/

# Промпты для редактирования изображений для каждого персонажа
IMAGE_EDIT_PROMPTS = {
    "cheb": "Удали фон у всех людей и фигур на фото и помести их в мир мультика Чебурашка. Фон должен быть уютным и добрым, в стиле советского мультфильма про Чебурашку. При необходимости перемести фигуры так, чтобы они стояли на полу и были вписаны в окружение. Чебурашка на фото уже есть, не добавляй еще одного.",
    "gena": "Удали фон у всех людей и фигур на фото и помести их в мир Крокодила Гены из кукольного мультфильма Чебурашка. Объемная кукольная анимация с картонными декорациями. Зоопарк или вечерние или утренние  улицы кукольного советского города. Красивое освещение, не тусклая картинка. Интеллигентная, слегка меланхоличная атмосфера. Гена носит костюм и шляпу, работает в зоопарке. Можно показать голубой вагон. При необходимости перемести фигуры так, чтобы они стояли на полу и были вписаны в окружение. Крокодил Гена на фото уже есть, не добавляй еще одного. Чебурашку тоже не добавляй.",
    "shap": "Удали фон у всех людей и фигур на фото и помести их в мир Старухи Шапокляк из кукольного мультфильма Чебурашка. Объемная кукольная анимация с картонными декорациями. Темные подъезды, дворы, заброшенные уголки. Контрастное освещение, драматичные серо-коричневые тона с яркими акцентами. Озорная хулиганская атмосфера приключений и проказ. При необходимости перемести фигуры так, чтобы они стояли на полу и были вписаны в окружение. Старуха Шапокляк на фото уже есть, не добавляй еще одну.",
    "volc": "Удали фон у всех людей и фигур на фото и помести их в мир рисованного мультфильма Ну, погоди! Классическая рисованная советская анимация. Фон должен быть динамичным и ярким, в стиле советского мультфильма. Яркие насыщенные цвета, четкие черные контуры, плоскостная графика. Советская среда. При необходимости перемести фигуры так, чтобы они стояли на полу и были вписаны в окружение. Но обязательно чтобы выглядело хорошо. Волк на фото уже есть, не добавляй еще одного."
}

RVC_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

TTS_VOICES = {
    "cheb": "alena",
    "gena": "ermil",
    "shap": "jane",
    "volc": "filipp"
}

RVC_MODELS = {
    "cheb": {"model": "cheb", "has_index": True},
    "gena": {"model": "gena", "has_index": True},
    "shap": {"model": "shap", "has_index": True},
    "volc": {"model": "volc", "has_index": False}
}

SYSTEM_PROMPTS = {
    "cheb": "Ты — Чебурашка. Отвечай дружелюбно, коротко и по-детски.",
    "gena": "Ты — Крокодил Гена. Отвечай вежливо, рассудительно и немного меланхолично, как в мультфильме. Обращайся к собеседнику 'мой друг'.",
    "shap": "Ты — Старуха Шапокляк. Отвечай вредно, с сарказмом, поучай и иногда хихикай.",
    "volc": "Ты — Волк из 'Ну, погоди!'. Отвечай немного грубовато, но с юмором, и можешь в конце добавить 'Ну, Заяц, погоди!'"
}

if os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

INDEX_PATH = "index.html"

sessions = {}

def add_to_history(device_id: str, character: str, role: str, text: str):
    """Добавить сообщение в историю конкретного персонажа"""
    if device_id not in sessions:
        sessions[device_id] = {}
    
    if character not in sessions[device_id]:
        sessions[device_id][character] = []
    
    sessions[device_id][character].append({"role": role, "text": text})
    
    if len(sessions[device_id][character]) > 10:
        sessions[device_id][character].pop(0)

def get_history(device_id: str, character: str) -> list:
    """Получить историю конкретного персонажа"""
    if device_id not in sessions or character not in sessions[device_id]:
        return []
    return sessions[device_id][character]

# ==================== HELPER FUNCTIONS ====================

def run_cmd(cmd):
    """Run shell command, raise on error, return stdout"""
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("ERROR:", proc.stderr.decode('utf-8', errors='ignore'))
        raise RuntimeError("Command failed")
    return proc.stdout

def webm_to_ogg(input_path: str, out_ogg: str, sample_rate=48000):
    """Конвертируем webm -> OGG Opus для Yandex STT через ffmpeg."""
    print("Converting", input_path, "->", out_ogg)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "libopus",
        "-ar", str(sample_rate),
        "-ac", "1",
        out_ogg
    ]
    run_cmd(cmd)
    print("Converted to", out_ogg)

def rvc_convert_infer(
    input_audio: str,
    output_audio: str,
    model_path: str,
    index_path: str = None,
    f0_up_key: int = 0,
    f0_method: str = "pm",
    index_rate: float = 0.75,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33
):
    """RVC конвертация через RvcWebUI localhost API."""
    try:
        model_name = model_path
        
        # 1. Очистка кеша RVC
        # print("Очищаем кеш RVC...")
        # response = requests.post("http://localhost:7897/run/infer_clean", json={
        #     "data": []
        # }, timeout=30)
        
        # if response.status_code != 200 or response.json().get('data') is None:
        #     print("Предупреждение: не удалось очистить кеш RVC")
        # else:
        #     print("Кеш успешно очищен")
        
        # 2. Выбор голоса (модели)

        print(f"Выбираем голос: {model_name}.pth")
        response = requests.post("http://localhost:7897/run/infer_set", json={
            "data": [
                f"{model_name}.pth",
                protect,
                protect
            ]
        }, timeout=30)
        
        if response.status_code != 200 or response.json().get('data') is None:
            raise RuntimeError(f"Не удалось выбрать голос {model_name}")
        
        print("Голос успешно выбран")
        
        print(f"Запускаем переозвучку через RvcWebUI...")
        
        index_file_path = f"logs/{model_name}.index" if index_path else ""
        
        response = requests.post("http://localhost:7897/run/infer_convert", json={
            "data": [
                f0_up_key,
                input_audio,
                0,
                None,
                f0_method,
                "",
                index_file_path,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
            ]
        }, timeout=120)
        
        if response.status_code != 200 or response.json().get('data') is None:
            raise RuntimeError("Не удалось выполнить переозвучку")
        
        result_data = response.json()["data"]
        revoiced_path = result_data[1]["name"]
        
        print(f"Переозвучка завершена: {revoiced_path}")
        
        shutil.copy(revoiced_path, output_audio)
        
        return output_audio
        
    except Exception as e:
        print(f"Error in RVC conversion: {e}")
        raise

def send_to_nano_banana(image_base64: str, prompt: str) -> str:
    """
    Отправляет изображение в Nano Banana через OpenRouter для редактирования.
    Возвращает base64 отредактированного изображения.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash-image",  # Платная модель
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    print(f"🤖 Отправка в Nano Banana с промптом: {prompt[:100]}...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ Ошибка API: {response.status_code}")
            print(f"Ответ: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"OpenRouter API error: {response.text}")
        
        result = response.json()
        
        # Извлекаем base64 изображения из ответа
        # Формат: choices[0].message.images[0].image_url.url
        if "choices" in result and result["choices"]:
            message = result["choices"][0]["message"]
            
            # Проверяем наличие images в ответе
            if "images" in message and message["images"]:
                image_url = message["images"][0]["image_url"]["url"]
                
                # Формат: data:image/png;base64,<данные>
                if "base64," in image_url:
                    base64_data = image_url.split("base64,")[1]
                    print("✅ Изображение успешно получено от Nano Banana")
                    return base64_data
                else:
                    print("❌ Неверный формат image_url")
                    raise HTTPException(status_code=500, detail="Invalid image URL format")
            else:
                print("❌ В ответе нет images")
                print(f"Структура ответа: {message.keys()}")
                raise HTTPException(status_code=500, detail="No images in API response")
        else:
            print("❌ Неверный формат ответа API")
            raise HTTPException(status_code=500, detail="Invalid API response format")
            
    except requests.exceptions.Timeout:
        print("❌ Таймаут при обращении к API")
        raise HTTPException(status_code=504, detail="API timeout")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    if not os.path.exists(INDEX_PATH):
        return HTMLResponse(content="index.html not found", status_code=404)
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/chat-stream")
async def chat_stream_endpoint(
    audio: UploadFile = File(...),
    character: str = Form(...),
    device_id: str = Form(...)
):
    """Полный цикл: STT -> Chat -> TTS -> RVC для конкретного персонажа"""
    
    # Читаем аудио заранее, до генератора
    audio_data = await audio.read()
    
    async def generate_response():
        # Начало общего отсчёта времени
        total_start_time = time.time()
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # 1. Сохраняем загруженный аудиофайл
                audio_path = os.path.join(temp_dir, "input.webm")
                with open(audio_path, "wb") as f:
                    f.write(audio_data)
                
                # Получаем системный промпт для выбранного персонажа
                system_prompt = SYSTEM_PROMPTS.get(character, SYSTEM_PROMPTS["cheb"])
                
                # 2. Конвертируем webm -> ogg для STT
                print("\n[ЭТАП 1/5] Конвертация WebM -> OGG для STT")
                stage_start = time.time()
                
                audio_ogg = os.path.join(temp_dir, "input.ogg")
                webm_to_ogg(audio_path, audio_ogg)
                
                stage_time = time.time() - stage_start
                print(f"⏱️  Конвертация WebM->OGG: {stage_time:.2f} сек")
                
                # 3. Speech-to-Text
                print("\n[ЭТАП 2/5] Распознавание речи (STT)")
                stage_start = time.time()
                
                with open(audio_ogg, "rb") as f:
                    stt_response = await client.post(
                        f"https://stt.api.cloud.yandex.net/speech/v1/stt:recognize?lang=ru-RU&folderId={FOLDER_ID}&format=oggopus",
                        headers={"Authorization": f"Api-Key {API_KEY}"},
                        content=f.read()
                    )
                
                stage_time = time.time() - stage_start
                print(f"⏱️  STT (распознавание речи): {stage_time:.2f} сек")
                
                if stt_response.status_code != 200:
                    raise HTTPException(status_code=stt_response.status_code, detail=stt_response.text)
                
                user_text = stt_response.json().get("result", "")
                if not user_text:
                    raise HTTPException(status_code=400, detail="Не удалось распознать речь")
                
                print(f"Recognized text: {user_text}")
                
                # ОТПРАВЛЯЕМ ПЕРВЫЙ CHUNK: user_text сразу после STT
                yield json.dumps({"type": "stt", "user_text": user_text}) + "\n"
                
                # 4. Генерируем ответ через YandexGPT с динамическим промптом
                # 4. Отправка в LLM через OpenRouter
                print("\n[ЭТАП 3/5] Генерация ответа (LLM)")
                stage_start = time.time()

                system_prompt = SYSTEM_PROMPTS.get(character, "Ты — дружелюбный помощник.")
                
                messages = [{"role": "system", "text": system_prompt}]

                # Добавляем историю переписки
                history = get_history(device_id, character)
                messages.extend(history)

                # Добавляем новое сообщение пользователя
                messages.append({"role": "user", "text": user_text})

                # 4. Генерируем ответ через YandexGPT с динамическим промптом
                chat_response = await client.post(
                    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                    headers={"Authorization": f"Api-Key {API_KEY}", "Content-Type": "application/json"},
                    json={
                        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt-lite/latest",
                        "completionOptions": {"stream": False, "temperature": 0.6, "maxTokens": "200"},

                        "messages": messages  
                    }
                )

                stage_time = time.time() - stage_start
                print(f"⏱️  LLM (генерация ответа): {stage_time:.2f} сек")
                
                if chat_response.status_code != 200:
                    raise HTTPException(status_code=chat_response.status_code, detail=chat_response.text)
                
                reply_text = chat_response.json()["result"]["alternatives"][0]["message"]["text"]

                add_to_history(device_id, character, "user", user_text)
                add_to_history(device_id, character, "assistant", reply_text)

                print(f"Generated reply: {reply_text}")
                
                # 5. Text-to-Speech
                print("\n[ЭТАП 4/5] Синтез речи (TTS)")
                stage_start = time.time()
                
                selected_voice = TTS_VOICES.get(character, "alena")
                tts_response = await client.post(
                    "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize",
                    headers={"Authorization": f"Api-Key {API_KEY}"},
                    data={
                        "text": reply_text, "lang": "ru-RU", "voice": selected_voice,
                        "folderId": FOLDER_ID, "format": "oggopus", "sampleRateHertz": "48000"
                    }
                )

                stage_time = time.time() - stage_start
                print(f"⏱️  TTS (синтез речи): {stage_time:.2f} сек")

                if tts_response.status_code != 200:
                    raise HTTPException(status_code=tts_response.status_code, detail=tts_response.text)
                
                tts_ogg = os.path.join(temp_dir, "tts.ogg")
                with open(tts_ogg, "wb") as f:
                    f.write(tts_response.content)
                
                # 7. Конвертируем OGG в WAV для RVC
                print("\n[ЭТАП 5/5] Клонирование голоса (RVC)")
                stage_start = time.time()
                
                tts_wav = os.path.join(temp_dir, "tts.wav")
                run_cmd(["ffmpeg", "-y", "-i", tts_ogg, "-ar", "40000", "-ac", "1", tts_wav])
                
                # 8. Применяем RVC с динамической моделью
                final_audio = tts_wav
                model_config = RVC_MODELS.get(character)

                if model_config:
                    rvc_out = os.path.join(temp_dir, "rvc_out.wav")
                    try:
                        model_name = model_config["model"]
                        index_name = model_name if model_config.get("has_index", False) else None
                        
                        final_audio = rvc_convert_infer(
                            input_audio=tts_wav, 
                            output_audio=rvc_out,
                            model_path=model_name,
                            index_path=index_name,  # None для volc
                            f0_up_key=0, 
                            f0_method="pm", 
                            index_rate=0.85
                        )
                    except Exception as e:
                        print(f"RVC failed for {character}, using TTS output: {e}")
                else:
                    print(f"RVC model for '{character}' not found. Using original TTS.")

                stage_time = time.time() - stage_start
                print(f"⏱️  RVC (клонирование голоса): {stage_time:.2f} сек")

                # 9. Конвертируем финальный результат в OGG
                final_mp3 = os.path.join(temp_dir, "final.mp3")
                run_cmd(["ffmpeg", "-y", "-i", final_audio, "-acodec", "libmp3lame", "-b:a", "128k", final_mp3])
                
                # 10. Читаем и кодируем в base64
                with open(final_mp3, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode('utf-8')
                
                # Подсчёт общего времени
                total_time = time.time() - total_start_time
                print(f"\n{'='*60}")
                print(f"🎯 ОБЩЕЕ ВРЕМЯ ОБРАБОТКИ: {total_time:.2f} сек")
                print(f"{'='*60}\n")
                
                # ОТПРАВЛЯЕМ ВТОРОЙ CHUNK: финальный ответ с аудио
                yield json.dumps({"type": "final", "reply_text": reply_text, "audio_base64": audio_b64}) + "\n"
    
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    return StreamingResponse(generate_response(), media_type="application/x-ndjson")

@app.post("/remove")
async def remove_background(
    photo: UploadFile = File(...),
    ar_overlay: UploadFile = File(...),
    active_target: str = Form(None)
):
    """
    НОВАЯ ЛОГИКА:
    1. Накладываем AR-слой поверх обычного фото (если есть AR-контент)
    2. Отправляем композитное изображение в Nano Banana
    3. Получаем отредактированное изображение
    """
    # Проверка типов файлов
    if photo.content_type.split('/')[0] != "image":
        raise HTTPException(status_code=400, detail="Photo file is not an image")
    if ar_overlay.content_type.split('/')[0] != "image":
        raise HTTPException(status_code=400, detail="AR overlay file is not an image")

    photo_data = await photo.read()
    ar_data = await ar_overlay.read()
    
    os.makedirs("debug", exist_ok=True)
    with open("debug/input_photo.png", "wb") as f: f.write(photo_data)
    with open("debug/input_ar.png", "wb") as f: f.write(ar_data)
    print(f"✓ Входящие изображения сохранены, активный маркер: {active_target}")
    
    try:
        # 1. Открываем обычное фото
        photo_image = Image.open(io.BytesIO(photo_data)).convert("RGBA")
        
        # 2. Открываем AR-слой
        ar_image = Image.open(io.BytesIO(ar_data)).convert("RGBA")
        # Проверяем, есть ли непрозрачные пиксели (альфа-канал > 0)
        has_ar_content = False
        # Проверяем, есть ли непрозрачные пиксели в AR-слое
        for pixel in ar_image.getdata():
            if pixel[3] > 0:  # Если альфа-канал > 0, то пиксель не прозрачный
                has_ar_content = True
                break
        
        # 3. Накладываем AR-слой поверх фото
        composite = photo_image.copy()
        
        if has_ar_content:
            print("✓ AR-контент обнаружен, накладываем поверх фото")
            # Масштабируем AR оверлей если нужно
            if ar_image.size != photo_image.size:
                ar_image = ar_image.resize(photo_image.size, Image.Resampling.LANCZOS)
            composite.paste(ar_image, (0, 0), ar_image)
        else:
            print("⚠ AR-контент отсутствует, используем только фото")
        
        # Сохраняем композит для отладки
        composite.save("debug/composite_before_ai.png", format="PNG")
        print("✓ Композитное изображение сохранено для отладки")
        
        # 4. Конвертируем композитное изображение в base64
        buffer = io.BytesIO()
        composite.save(buffer, format="PNG")
        composite_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 5. Выбираем промпт в зависимости от персонажа
        bg_options = ['cheb', 'volc', 'gena', 'shap']
        selected_character = active_target if active_target and active_target in bg_options else random.choice(bg_options)
        prompt = IMAGE_EDIT_PROMPTS.get(selected_character, IMAGE_EDIT_PROMPTS["cheb"])
        
        print(f"✓ Выбран персонаж: {selected_character}")
        print(f"✓ Промпт: {prompt}")
        
        # 6. Отправляем в Nano Banana
        print("🚀 Отправка в Nano Banana...")
        result_base64 = send_to_nano_banana(composite_base64, prompt)
        
        # 7. Декодируем результат
        result_bytes = base64.b64decode(result_base64)
        result_image = Image.open(io.BytesIO(result_bytes))
        
        # Сохраняем результат под уникальным именем с текущим временем
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"debug/output_result_{timestamp}.png"
        result_image.save(output_path, format="PNG")
        print(f"✓ Результат сохранен: {output_path}")
        
        # 8. Возвращаем результат
        out_io = io.BytesIO()
        result_image.save(out_io, format="PNG")
        out_io.seek(0)
        
        return StreamingResponse(out_io, media_type="image/png")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=443, 
        reload=True, 
        ssl_keyfile=r"C:\Certbot\live\vrkodex.ru\privkey.pem", 
        ssl_certfile=r"C:\Certbot\live\vrkodex.ru\fullchain.pem"
    )

# if __name__ == "__main__":
#     uvicorn.run(
#         "server:app", 
#         host="0.0.0.0", 
#         port=8000, 
#         reload=True,
#         reload_excludes=["venv/*"]
#     )
