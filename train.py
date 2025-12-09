
import os
from ultralytics import YOLO

def main():
    print("🚗 Обучение YOLO для номерных знаков")
    print("=" * 40)

    # 1. Проверяем датасет
    print("\n1. Проверка датасета...")

    required = [
        "datasets/images/train",
        "datasets/images/val",
        "datasets/labels/train",
        "datasets/labels/val"
    ]

    for folder in required:
        if not os.path.exists(folder):
            print(f"❌ Нет папки: {folder}")
            return

    # Считаем файлы
    train_images = len([f for f in os.listdir("datasets/images/train") if f.endswith(('.jpg', '.png'))])
    val_images = len([f for f in os.listdir("datasets/images/val") if f.endswith(('.jpg', '.png'))])

    train_labels = len([f for f in os.listdir("datasets/labels/train") if f.endswith('.txt')])
    val_labels = len([f for f in os.listdir("datasets/labels/val") if f.endswith('.txt')])

    print(f"✅ Train: {train_images} изображений, {train_labels} аннотаций")
    print(f"✅ Val: {val_images} изображений, {val_labels} аннотаций")

    # 2. Создаем datasets.yaml
    print("\n2. Создание конфигурации...")

    yaml_content = """path: datasets
train: images/train
val: images/val
names:
  0: license_plate
nc: 1"""

    with open('datasets.yaml', 'w') as f:
        f.write(yaml_content)

    print("✅ Создан datasets.yaml")

    # 3. Загружаем модель
    print("\n3. Загрузка модели...")

    try:
        model = YOLO('yolo11n.pt')
        print("✅ YOLO11n загружена")
    except:
        print("📥 Скачиваю YOLO11n...")
        model = YOLO('yolo11n.pt')

    # 4. Обучаем
    print("\n4. Начало обучения...")
    print("   Это займет некоторое время...")

    model.train(
        data='datasets.yaml',
        epochs=20,
        imgsz=640,
        batch=8,
        save=True,
        project='plate_training',
        name='yolo11n'
    )

    print("\n✅ Обучение завершено!")
    print("📁 Результаты в папке: plate_training/yolo11n/")
    print("📄 Модель: plate_training/yolo11n/weights/best.pt")

if __name__ == "__main__":
    main()