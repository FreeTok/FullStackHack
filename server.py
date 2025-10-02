# server.py
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rembg import remove
from PIL import Image
import uvicorn
import os

app = FastAPI(title="Background remover")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

INDEX_PATH = "index.html"

@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH, media_type="text/html")
    return HTMLResponse("<h3>Place index.html in the same folder and refresh.</h3>")

@app.post("/remove")
async def remove_background(
    photo: UploadFile = File(...),
    ar_overlay: UploadFile = File(...),
    active_target: str = Form(None)
):
    # Проверка типов файлов
    if photo.content_type.split('/')[0] != "image":
        raise HTTPException(status_code=400, detail="Photo file is not an image")
    if ar_overlay.content_type.split('/')[0] != "image":
        raise HTTPException(status_code=400, detail="AR overlay file is not an image")

    # Читаем оба изображения
    photo_data = await photo.read()
    ar_data = await ar_overlay.read()
    
    # Сохраняем для отладки
    os.makedirs("debug", exist_ok=True)
    with open("debug/input_photo.png", "wb") as f:
        f.write(photo_data)
    with open("debug/input_ar.png", "wb") as f:
        f.write(ar_data)
    print(f"✓ Входящие изображения сохранены, активный маркер: {active_target}")
    
    try:
        # Открываем фото
        photo_image = Image.open(io.BytesIO(photo_data)).convert("RGBA")
        
        # Удаляем фон с фото
        photo_bytes = io.BytesIO()
        photo_image.save(photo_bytes, format="PNG")
        photo_bytes = photo_bytes.getvalue()
        
        photo_no_bg_bytes = remove(photo_bytes)
        photo_no_bg = Image.open(io.BytesIO(photo_no_bg_bytes)).convert("RGBA")
        
        # Открываем AR-проекцию
        ar_image = Image.open(io.BytesIO(ar_data)).convert("RGBA")

        # Проверяем, есть ли реальный AR-контент
        ar_pixels = ar_image.getdata()
        has_ar_content = any(pixel[3] > 0 for pixel in ar_pixels)

        if not has_ar_content:
            print("⚠ AR-контент отсутствует, используем только фото без фона")

        print(active_target)

        # Загружаем соответствующий фон в зависимости от активного маркера
        background = None
        bg_options = ['cheb', 'volc', 'gena', 'shap']

        if active_target and active_target in bg_options:
            bg_path = f"assets/bg_{active_target}.png"
            if os.path.exists(bg_path):
                background = Image.open(bg_path).convert("RGBA")
        else:
            # Выбираем случайный фон если маркер не определен
            import random
            random_target = random.choice(bg_options)
            bg_path = f"assets/bg_{random_target}.png"
            if os.path.exists(bg_path):
                background = Image.open(bg_path).convert("RGBA")
                print(f"✓ Выбран случайный фон: {random_target}")

        # Создаём итоговое изображение
        if background:
            bg_ratio = background.width / background.height
            photo_ratio = photo_no_bg.width / photo_no_bg.height
            
            if bg_ratio > photo_ratio:
                new_height = photo_no_bg.height
                new_width = int(new_height * bg_ratio)
                background = background.resize((new_width, new_height), Image.Resampling.LANCZOS)
                left = (new_width - photo_no_bg.width) // 2
                background = background.crop((left, 0, left + photo_no_bg.width, new_height))
            else:
                new_width = photo_no_bg.width
                new_height = int(new_width / bg_ratio)
                background = background.resize((new_width, new_height), Image.Resampling.LANCZOS)
                top = new_height - photo_no_bg.height
                background = background.crop((0, top, new_width, new_height))
            
            result = background
        else:
            # Если фона нет, используем белый фон
            result = Image.new("RGBA", photo_no_bg.size, (255, 255, 255, 255))

        # Накладываем фото без фона
        result.paste(photo_no_bg, (0, 0), photo_no_bg)

        # Накладываем AR-проекцию поверх ТОЛЬКО если есть контент
        if has_ar_content:
            # Масштабируем AR-проекцию с сохранением пропорций и качества
            if ar_image.size != photo_no_bg.size:
                scale = min(photo_no_bg.width / ar_image.width, 
                            photo_no_bg.height / ar_image.height)
                new_size = (int(ar_image.width * scale), int(ar_image.height * scale))
                ar_image = ar_image.resize(new_size, Image.Resampling.LANCZOS)
                
                ar_canvas = Image.new("RGBA", photo_no_bg.size, (0, 0, 0, 0))
                offset = ((photo_no_bg.width - ar_image.width) // 2,
                        (photo_no_bg.height - ar_image.height) // 2)
                ar_canvas.paste(ar_image, offset, ar_image)
                ar_image = ar_canvas

            result.paste(ar_image, (0, 0), ar_image)   
     
        # Масштабируем AR-проекцию с сохранением пропорций и качества
        if ar_image.size != photo_no_bg.size:
            # Вычисляем масштаб
            scale = min(photo_no_bg.width / ar_image.width, 
                        photo_no_bg.height / ar_image.height)
            new_size = (int(ar_image.width * scale), int(ar_image.height * scale))
            
            # Используем LANCZOS для максимального качества
            ar_image = ar_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Создаем прозрачный canvas
            ar_canvas = Image.new("RGBA", photo_no_bg.size, (0, 0, 0, 0))
            # Центрируем
            offset = ((photo_no_bg.width - ar_image.width) // 2,
                    (photo_no_bg.height - ar_image.height) // 2)
            ar_canvas.paste(ar_image, offset, ar_image)
            ar_image = ar_canvas

        # Накладываем AR-проекцию поверх
        result.paste(ar_image, (0, 0), ar_image)
        
        # Сохраняем результат для отладки
        result.save("debug/output_result.png", format="PNG")
        print("✓ Результат сохранен: debug/output_result.png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    # Возвращаем результат
    out_io = io.BytesIO()
    result.save(out_io, format="PNG")
    out_io.seek(0)
    return StreamingResponse(out_io, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=443, reload=True, ssl_keyfile=r"C:\Certbot\live\vrkodex.ru\privkey.pem", ssl_certfile=r"C:\Certbot\live\vrkodex.ru\fullchain.pem")