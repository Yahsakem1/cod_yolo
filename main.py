from src.split import run_split
from src.split_image import copy_images, SPLITS
from src.convert import coco_to_yolo, JSON_TRAIN, JSON_VAL, JSON_TEST
import os


def main():
    print("=== 1. Split ===")
    run_split()

    print("\n=== 2. Copy images ===")
    for name, (json_path, src_dir) in SPLITS.items():
        copy_images(json_path, src_dir, name)

    print("\n=== 3. COCO -> YOLO ===")
    coco_to_yolo(JSON_TRAIN, "train")
    coco_to_yolo(JSON_VAL, "val")
    if os.path.exists(JSON_TEST):
        coco_to_yolo(JSON_TEST, "test")
    else:
        print(f"[!] Test JSON не найден: {JSON_TEST}")


if __name__ == "__main__":
    main()
