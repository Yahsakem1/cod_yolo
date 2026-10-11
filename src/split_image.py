import json
import shutil
import os
# ==========================
# Настройки
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
DEST_IMAGES_DIR = os.path.join(DATA_DIR, "images")
SOURCE_TRAIN_IMAGES = os.path.join(DATA_DIR, "train", "Image")
SOURCE_TEST_IMAGES = os.path.join(DATA_DIR, "test", "Image")
SPLITS = {
    "train": (os.path.join(ANNOTATIONS_DIR, "train_split.json"), SOURCE_TRAIN_IMAGES),
    "val":   (os.path.join(ANNOTATIONS_DIR, "val_split.json"),   SOURCE_TRAIN_IMAGES),
    "test":  (os.path.join(ANNOTATIONS_DIR, "CAM_Instance_Test.json"),        SOURCE_TEST_IMAGES),  
}
# ==========================
# Копирование
# ==========================
def copy_images(json_path, source_dir, split_name):
    if not os.path.exists(json_path):
        print(f"[!] {split_name}: JSON не найден по пути {json_path} — пропущено")
        return
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)
    dest_dir = os.path.join(DEST_IMAGES_DIR, split_name)
    os.makedirs(dest_dir, exist_ok=True)
    missing = []
    for img in coco["images"]:
        fname = img["file_name"]
        src = os.path.join(source_dir, fname)
        dst = os.path.join(dest_dir, fname)
        if not os.path.exists(src):
            missing.append(fname)
            continue
        shutil.copy2(src, dst)
    copied = len(coco["images"]) - len(missing)
    print(f"{split_name}: скопировано {copied}, не найдено {len(missing)}")
    if missing:
        print("  Примеры отсутствующих:", missing[:5])
# ==========================
# Запуск
# ==========================
if __name__ == "__main__":
    for split_name, (json_path, source_dir) in SPLITS.items():
        copy_images(json_path, source_dir, split_name)