import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    print("🚀 Запуск системы распознавания автомобильных номеров")
    print("🤖 Используется TrOCR (Transformer OCR) от Facebook")
    print("📖 Модель предварительно обучена на печатном тексте")
    print("🌐 Откройте в браузере: http://localhost:8000")

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )