import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
import src.boundary_head_trainer  
from src.freq_c2f import FreqC2f
tasks.FreqC2f = FreqC2f
WEIGHTS = PROJECT / "runs" / "cod" / "v8s_fft_p4_boundary_head-4" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT / "runs" / "predict"
def main():
    if not WEIGHTS.exists():
        print(f"Веса не найдены: {WEIGHTS}")
        return

    print("Загрузка модели...")
    model = YOLO(str(WEIGHTS))

    raw_path = input("Введите путь к изображению: ").strip().strip('"').strip("'")
    img_path = Path(raw_path)

    if not img_path.exists():
        print(f"Файл не найден: {img_path}")
        return

    results = model.predict(
        source=str(img_path),
        imgsz=640,
        conf=0.25,
        retina_masks=True,
        save=True,
        project=str(OUTPUT_DIR),
        name="v8s_fft_p4_boundary_head-4",
        exist_ok=True,
    )

    save_dir = Path(results[0].save_dir)
    print(f"Готово! Результат сохранён в: {save_dir / img_path.name}")
if __name__ == "__main__":
    main()