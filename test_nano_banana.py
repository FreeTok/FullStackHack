#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы Nano Banana через OpenRouter
"""

import requests
import base64
from PIL import Image
from io import BytesIO
import sys

# ВСТАВЬ СВОЙ API КЛЮЧ СЮДА
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxx"

def image_to_base64(image_path):
    """Конвертирует изображение в base64"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def send_to_nano_banana(image_base64, prompt):
    """Отправляет изображение в Nano Banana для редактирования"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash-image",
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
    
    print(f"🚀 Отправка в Nano Banana...")
    print(f"📝 Промпт: {prompt}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ Ошибка API: {response.status_code}")
            print(f"Ответ: {response.text}")
            return None
        
        result = response.json()
        
        # Извлекаем base64 изображения из ответа
        # Формат: choices[0].message.images[0].image_url.url
        if "choices" in result and result["choices"]:
            message = result["choices"][0]["message"]
            
            if "images" in message and message["images"]:
                image_url = message["images"][0]["image_url"]["url"]
                
                if "base64," in image_url:
                    base64_data = image_url.split("base64,")[1]
                    print("✅ Изображение успешно получено!")
                    return base64_data
                else:
                    print("❌ Неверный формат image_url")
                    return None
            else:
                print("❌ В ответе нет images")
                print(f"Ответ: {message}")
                return None
        else:
            print("❌ Неверный формат ответа API")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Таймаут при обращении к API")
        return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

def test_nano_banana(input_image_path, output_image_path, prompt):
    """Тестирует Nano Banana на изображении"""
    
    print("="*60)
    print("🍌 ТЕСТ NANO BANANA")
    print("="*60)
    
    # 1. Читаем изображение
    print(f"\n📂 Читаем изображение: {input_image_path}")
    try:
        image = Image.open(input_image_path).convert("RGBA")
        print(f"✅ Изображение загружено: {image.size}")
    except Exception as e:
        print(f"❌ Ошибка при чтении изображения: {e}")
        return False
    
    # 2. Конвертируем в base64
    print(f"\n🔄 Конвертация в base64...")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print(f"✅ Конвертировано ({len(image_base64)} символов)")
    
    # 3. Отправляем в Nano Banana
    print(f"\n🌐 Отправка в Nano Banana...")
    result_base64 = send_to_nano_banana(image_base64, prompt)
    
    if not result_base64:
        print("❌ Не удалось получить результат")
        return False
    
    # 4. Сохраняем результат
    print(f"\n💾 Сохранение результата...")
    try:
        result_bytes = base64.b64decode(result_base64)
        result_image = Image.open(BytesIO(result_bytes))
        result_image.save(output_image_path)
        print(f"✅ Результат сохранен: {output_image_path}")
        print(f"📐 Размер: {result_image.size}")
    except Exception as e:
        print(f"❌ Ошибка при сохранении: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 ТЕСТ УСПЕШНО ЗАВЕРШЕН!")
    print("="*60)
    return True

if __name__ == "__main__":
    # Проверка аргументов
    if len(sys.argv) < 2:
        print("Использование: python test_nano_banana.py <путь_к_изображению> [выходной_файл]")
        print("\nПример:")
        print("  python test_nano_banana.py photo.jpg output.png")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "nano_banana_result.png"
    
    # Тестовый промпт
    prompt = """Удали фон у всех людей и фигур на фото и помести их в мир советского мультика про Чебурашку. 
Фон должен быть уютным и добрым, в стиле классической советской анимации."""
    
    # Запускаем тест
    success = test_nano_banana(input_path, output_path, prompt)
    
    sys.exit(0 if success else 1)
