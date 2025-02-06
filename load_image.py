import asyncio
import concurrent.futures
from pathlib import Path
from collections import defaultdict
import pandas as pd
import cv2
import numpy as np
import torch
from aiohttp import ClientSession
from ultralytics import YOLO

# Пути к файлам
BASE_DIR: Path = Path(__file__).parent
RESULT_PATH: Path = BASE_DIR / "results/"
ANNOTATED_PATH = RESULT_PATH / "annotated/"

# Колонки
IMAGE_COLUMNS = [
    "Фото Входная группа/фасад_МФ+Yota+конкур",
    "Фото Ресепшн/Зона касс_МФ+Yota+конкур",
    "Фото Витрины_МФ+Yota+конкур",
    "Фото Интерьер_МФ+Yota+конкур",
]

EVALUATION_COLUMNS = [
    "Входная зона МФ",
    "Входная зона Yota",
    "Ресепшен МФ",
    "Ресепшен Yota",
    "Витрины МФ",
    "Витрины Yota",
    "Интерьер МФ",
    "Интерьер Yota",
]


### === 1. Работа с файлами === ###
def load_data(filename: str, sheet_name: str) -> pd.DataFrame:
    """Загружает данные из Excel."""
    print(f"📂 Загружаем данные из {filename} (лист: {sheet_name})...")
    return pd.read_excel(filename, sheet_name=sheet_name)


def save_batch_to_excel(batch_df: pd.DataFrame, batch_num: int):
    """Сохраняет батч в отдельный файл."""
    batch_filename = RESULT_PATH / f"batch_{batch_num}.xlsx"
    batch_df.to_excel(batch_filename, index=False)
    print(f"✅ Сохранен батч {batch_num} -> {batch_filename}")


def combine_batches():
    """Объединяет все батчи в финальный Excel-файл."""
    print("🔄 Объединяем все батчи в единый файл...")
    all_batches = [pd.read_excel(f) for f in RESULT_PATH.glob("batch_*.xlsx")]
    final_df = pd.concat(all_batches, ignore_index=True)
    final_df.to_excel("final_output.xlsx", index=False)
    print("✅ Финальный файл сохранен как final_output.xlsx")


### === 2. Обработка изображений === ###
async def load_resized_image(
    session: ClientSession, url: str, sem: asyncio.Semaphore
) -> np.ndarray:
    """Загружает и изменяет размер изображения."""
    async with sem:
        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"Ошибка загрузки {url}: {response.status}")
            image_bytes = await response.read()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Ошибка декодирования изображения: {url}")
            return image


async def load_images_in_batch(batch_df: pd.DataFrame) -> dict:
    """Асинхронно загружает все изображения в батче и сохраняет их связь с колонками."""
    print("📸 Загружаем изображения для батча...")
    semaphore = asyncio.Semaphore(10)
    async with ClientSession() as session:
        tasks = {
            (idx, img_col): asyncio.create_task(
                load_resized_image(session, row[img_col], semaphore)
            )
            for idx, row in batch_df.iterrows()
            for img_col in IMAGE_COLUMNS
            if row[img_col] != "-"
        }

        images = {}
        for key, task in tasks.items():
            try:
                images[key] = await task
            except Exception as e:
                print(f"❌ Ошибка загрузки {key}: {e}")
                images[key] = None

        print(f"✅ Загружено {len(images)} изображений")
        return images


def detect_objects_for_batch(model_path: str, images: list[np.ndarray]) -> list[dict]:
    """Обрабатывает батч изображений в процессе."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)

    detections = model.predict(images, verbose=False)
    objects_info_list = []

    for i, result in enumerate(detections):
        objects_info = defaultdict(int)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            area_px = (x2 - x1) * (y2 - y1)
            objects_info[label] += area_px
        objects_info_list.append(objects_info)
    return objects_info_list


async def detect_objects_in_batch(model_path: str, images: dict) -> dict:
    """Обрабатывает изображения в батче с помощью YOLO в процессах."""
    print("🔄 Запускаем YOLO-обработку в процессах...")
    loop = asyncio.get_running_loop()

    batch_size = 8
    image_batches = [
        list(images.values())[i : i + batch_size]
        for i in range(0, len(images), batch_size)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        tasks = [
            loop.run_in_executor(executor, detect_objects_for_batch, model_path, batch)
            for batch in image_batches
        ]
        results = await asyncio.gather(*tasks)

    detections = {}
    image_keys = list(images.keys())
    for i, batch_results in enumerate(results):
        for j, objects_info in enumerate(batch_results):
            detections[image_keys[i * batch_size + j]] = objects_info

    print(f"✅ Обработано {len(detections)} изображений в YOLO")
    return detections


def process_detections(batch_df: pd.DataFrame, detections: dict):
    """Записывает оценки в правильные колонки."""
    print("📝 Применяем оценки к батчу...")
    for (idx, img_col), objects_info in detections.items():
        # Получаем соответствующие колонки для МФ и Yota
        eval_col_mf = EVALUATION_COLUMNS[IMAGE_COLUMNS.index(img_col) * 2]  # МФ
        eval_col_yota = EVALUATION_COLUMNS[IMAGE_COLUMNS.index(img_col) * 2 + 1]  # Yota

        batch_df.at[idx, eval_col_mf] = get_mark(objects_info, "Megafon")
        batch_df.at[idx, eval_col_yota] = get_mark(objects_info, "Yota")

    print("✅ Оценки записаны в существующие поля")


def get_mark(objects_info, operator: str) -> int:
    """Анализирует объекты и возвращает оценку присутствия заданного оператора связи."""
    other_operators = {"Bilain", "T2", "MTC"}
    operator_area = objects_info.get(operator, 0)
    other_areas = [objects_info.get(op, 0) for op in other_operators]

    if 0 < operator_area < 25_000:
        pixels = 2500
    elif 25_000 <= operator_area < 100_000:
        pixels = 5000
    else:
        pixels = 7500
        

    if any(area > operator_area + pixels for area in other_areas if area != 0):
        return 0

    if any(
        operator_area - pixels <= area <= operator_area + pixels
        for area in other_areas
        if area != 0
    ):
        return 1

    if operator_area == 0:
        return 0 if any(other_areas) else 1

    return 2


### === 3. Основной процесс === ###
async def process_batch(batch_df: pd.DataFrame, batch_num: int):
    """Обрабатывает один батч."""
    print(f"🚀 Начинаем обработку батча {batch_num}...")
    images = await load_images_in_batch(batch_df)
    detections = await detect_objects_in_batch("best.pt", images)
    process_detections(batch_df, detections)
    save_batch_to_excel(batch_df, batch_num)


async def main():
    """Основная асинхронная функция."""
    df = load_data("test_file.xlsx", "Где оценивать")
    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    ANNOTATED_PATH.mkdir(parents=True, exist_ok=True)

    batch_size = 100
    total_batches = (len(df) + batch_size - 1) // batch_size  # Количество батчей

    for batch_num, start in enumerate(range(0, len(df), batch_size), start=1):
        print(f"\n🚀 Обрабатываем батч {batch_num}/{total_batches}...")

        batch_df = df.iloc[start: start + batch_size].copy()
        await process_batch(batch_df, batch_num) 
    combine_batches()


if __name__ == "__main__":
    asyncio.run(main())
