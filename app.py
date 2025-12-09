from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
from pathlib import Path
import time
from detection import LicensePlateDetector

app = FastAPI(title="Детектирование и распознавание автомобильных номеров")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Путь к модели YOLO
MODEL_PATH = "C:/Users/ekate/PycharmProjects/cv1/plate_training/yolo11n/weights/best.pt"

# Файл для логов (можно изменить путь)
LOG_FILE = "license_plates_log.txt"

# Инициализация детектора
detector = None
try:
    detector = LicensePlateDetector(MODEL_PATH, LOG_FILE)
    print("✅ Детектор инициализирован успешно!")
    print(f"📝 Лог-файл: {LOG_FILE}")
except Exception as e:
    print(f"❌ Ошибка инициализации детектора: {e}")
    print("Установите зависимости: pip install ultralytics transformers torch torchvision")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("detection.html", {
        "request": request,
        "model_loaded": detector is not None,
        "classes": detector.plate_classes if detector else []
    })


@app.get("/logs")
async def view_logs(request: Request):
    """Страница для просмотра логов"""
    logs_content = ""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                logs_content = f.read()
        except Exception as e:
            logs_content = f"Ошибка чтения лог-файла: {e}"
    else:
        logs_content = "Лог-файл пока не создан"

    return templates.TemplateResponse("logs.html", {
        "request": request,
        "logs_content": logs_content,
        "log_file": LOG_FILE
    })


@app.post("/detect")
async def detect_plates(request: Request, file: UploadFile = File(...)):
    if detector is None:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": "Детектор не инициализирован! Установите зависимости."
        })

    try:
        start_time = time.time()

        # Читаем изображение
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return templates.TemplateResponse("result.html", {
                "request": request,
                "error": "Не удалось загрузить изображение"
            })

        height, width = image.shape[:2]

        # Получаем имя файла для логирования
        filename = file.filename or "unknown.jpg"

        # Детекция номеров (передаем имя файла для логирования)
        detected_plates = detector.detect_plates(image, filename)

        # Сохраняем обрезанные изображения номеров
        for i, plate in enumerate(detected_plates):
            plate_id = f"plate_{i}_{int(time.time())}_{os.urandom(2).hex()}"
            plate_image_url = detector.save_plate_crop(plate['crop'], plate_id)
            plate['plate_image_url'] = plate_image_url
            plate['plate_id'] = f"Номер {i + 1}"
            # Удаляем crop из данных для ответа
            if 'crop' in plate:
                del plate['crop']

        # Рисуем результаты
        result_image = detector.draw_detections(image, detected_plates)

        # Сохраняем итоговое изображение
        uploads_dir = "static/uploads"
        os.makedirs(uploads_dir, exist_ok=True)

        if file.filename:
            filename_stem = Path(file.filename).stem
            ext = Path(file.filename).suffix or ".jpg"
        else:
            filename_stem = "image"
            ext = ".jpg"

        timestamp = int(time.time())
        output_filename = f"result_{filename_stem}_{timestamp}{ext}"
        output_path = os.path.join(uploads_dir, output_filename)
        cv2.imwrite(output_path, result_image)

        # Статистика
        recognized = [p for p in detected_plates if p['plate_text']]

        stats = {
            "total_detected": len(detected_plates),
            "recognized": len(recognized),
            "recognition_rate": f"{(len(recognized) / len(detected_plates) * 100 if detected_plates else 0):.1f}%",
            "image_size": f"{width}x{height}",
            "filename": filename,
            "model": "YOLO + TrOCR",
            "processing_time": f"{(time.time() - start_time):.2f} сек",
            "log_file": LOG_FILE
        }

        return templates.TemplateResponse("result.html", {
            "request": request,
            "detected_plates": detected_plates,
            "image_url": f"/static/uploads/{output_filename}",
            "stats": stats,
            "model_info": f"YOLO + TrOCR на {detector.device}",
            "classes": detector.plate_classes if detector else [],
            "model_loaded": True
        })

    except Exception as e:
        import traceback
        print(f"Ошибка: {traceback.format_exc()}")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"Ошибка обработки: {str(e)}",
            "model_loaded": detector is not None
        })