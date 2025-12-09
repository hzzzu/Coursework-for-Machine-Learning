import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import os
from pathlib import Path
from datetime import datetime


class LicensePlateDetector:
    def __init__(self, yolo_model_path, log_file="license_plates_log.txt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Файл для логов
        self.log_file = log_file

        # Загрузка YOLO модели
        self.yolo_model = YOLO(yolo_model_path)
        print(f"✅ YOLO модель загружена на {self.device}!")

        # Загрузка TrOCR модели
        print("⏳ Загрузка модели TrOCR...")
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
        self.trocr_model.to(self.device)
        print("✅ TrOCR модель загружена!")

        # Получаем классы из модели
        self.plate_classes = self.yolo_model.names if hasattr(self.yolo_model, 'names') else ["license_plate"]

        # Словарь для замены латинских букв на кириллические
        self.latin_to_cyrillic = {
            'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
            'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
            'X': 'Х', 'Y': 'У'
        }

        # Паттерны для российских номеров
        self.plate_patterns = [
            r'[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}',  # Стандартный
        ]

    def preprocess_for_trocr(self, plate_image):
        """Предобработка изображения для TrOCR"""
        if len(plate_image.shape) == 3:
            if plate_image.shape[2] == 4:
                plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGRA2RGB)
            else:
                plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)

        # Конвертируем в PIL Image
        pil_image = Image.fromarray(plate_image)

        # Применяем предобработку TrOCR
        pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        return pixel_values

    def recognize_with_trocr(self, plate_image):
        """Распознавание текста с помощью TrOCR"""
        try:
            # Предобработка
            pixel_values = self.preprocess_for_trocr(plate_image)

            # Генерация текста
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values)

            # Декодирование
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Очистка текста
            cleaned_text = self.clean_plate_text(generated_text)

            return cleaned_text, 0.8  # Фиксированная высокая уверенность

        except Exception as e:
            print(f"Ошибка TrOCR: {e}")
            return "", 0.0

    def clean_plate_text(self, text):
        """Очищает текст номерного знака"""
        if not text:
            return ""

        # Приводим к верхнему регистру
        text = text.upper()

        # Убираем нежелательные символы
        allowed_chars = set('АВЕКМНОРСТУХABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        cleaned = ''.join(c for c in text if c in allowed_chars)

        # Заменяем латинские буквы на кириллические
        final_text = ''
        for char in cleaned:
            if char in self.latin_to_cyrillic:
                final_text += self.latin_to_cyrillic[char]
            else:
                final_text += char

        # Проверяем паттерны
        for pattern in self.plate_patterns:
            matches = re.findall(pattern, final_text)
            if matches:
                return matches[0]

        return final_text

    def log_plate_info(self, filename, plate_text, confidence, text_confidence):
        """Записывает информацию о распознанном номере в лог-файл"""
        try:
            # Получаем текущую дату и время
            now = datetime.now()
            date_str = now.strftime("%d.%m.%Y")
            time_str = now.strftime("%H:%M:%S")

            # Формируем строку для записи
            log_entry = f"[{date_str} {time_str}] Файл: {filename} | Номер: {plate_text} | Уверенность детекции: {confidence:.2%} | Уверенность OCR: {text_confidence:.2%}\n"

            # Открываем файл для добавления (если файла нет - создается)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)

            print(f"✅ Запись в лог-файл: {log_entry.strip()}")
            return True
        except Exception as e:
            print(f"❌ Ошибка записи в лог-файл: {e}")
            return False

    def detect_plates(self, image, filename="unknown", confidence_threshold=0.3):
        """Основная функция детекции номеров"""
        height, width = image.shape[:2]
        detected_plates = []

        # Детекция с помощью YOLO
        results = self.yolo_model(image, conf=confidence_threshold)

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    # Координаты
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # Имя класса
                    class_name = self.plate_classes.get(class_id, f"Номер {i + 1}") if isinstance(self.plate_classes,
                                                                                                  dict) else f"Номер {i + 1}"

                    # Вырезаем область
                    x1_int, y1_int, x2_int, y2_int = map(int, [x1, y1, x2, y2])

                    # Добавляем отступы
                    padding = 10
                    x1_crop = max(0, x1_int - padding)
                    y1_crop = max(0, y1_int - padding)
                    x2_crop = min(width, x2_int + padding)
                    y2_crop = min(height, y2_int + padding)

                    plate_crop = image[y1_crop:y2_crop, x1_crop:x2_crop]

                    # Распознаем текст
                    plate_text, text_confidence = "", 0.0
                    if plate_crop.size > 100:
                        # Увеличиваем для лучшего распознавания
                        plate_crop_large = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        plate_text, text_confidence = self.recognize_with_trocr(plate_crop_large)

                    # Логируем информацию о номере
                    if plate_text:  # Логируем только если номер распознан
                        self.log_plate_info(filename, plate_text, float(confidence), float(text_confidence))

                    detected_plates.append({
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "bbox": [x1_int, y1_int, x2_int, y2_int],
                        "plate_text": plate_text,
                        "text_confidence": float(text_confidence),
                        "crop": plate_crop
                    })

        return detected_plates

    def draw_detections(self, image, detections):
        """Рисует обнаружения на изображении"""
        result_image = image.copy()

        for idx, plate in enumerate(detections):
            x1, y1, x2, y2 = plate['bbox']
            color = (0, 255, 0) if idx % 2 == 0 else (0, 0, 255)

            # Рисуем bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)

            # Текст для отображения
            label = f"{plate['class_name']}: {plate['confidence']:.2f}"
            if plate['plate_text']:
                label += f" - {plate['plate_text']}"

            # Рисуем текст
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Фон для текста
            cv2.rectangle(result_image,
                          (x1, y1 - th - 10),
                          (x1 + tw, y1),
                          color, -1)

            # Текст
            cv2.putText(result_image, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        return result_image

    def save_plate_crop(self, plate_crop, plate_id, uploads_dir="static/uploads"):
        """Сохраняет обрезанное изображение номера"""
        if plate_crop.size > 0:
            os.makedirs(uploads_dir, exist_ok=True)
            plate_path = os.path.join(uploads_dir, f"{plate_id}.jpg")
            cv2.imwrite(plate_path, plate_crop)
            return f"/static/uploads/{plate_id}.jpg"
        return ""