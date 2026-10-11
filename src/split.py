import json
import random
import os

random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))    
DATA_DIR = os.path.join(PROJECT_DIR, "data")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
SOURCE_JSON = os.path.join(DATA_DIR, "Train", "CAM_Instance_Train.json")


def filter_coco(coco, keep_ids):
    return {
        "info": coco["info"],
        "licenses": coco["licenses"],
        "categories": coco["categories"],
        "images": [img for img in coco["images"] if img["id"] in keep_ids],
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in keep_ids],
    }


def run_split():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    with open(SOURCE_JSON, encoding="utf-8") as f:
        coco = json.load(f)

    image_ids = [img["id"] for img in coco["images"]]
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * 0.85)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    with open(os.path.join(ANNOTATIONS_DIR, "train_split.json"), "w", encoding="utf-8") as f:
        json.dump(filter_coco(coco, train_ids), f, indent=4)
    with open(os.path.join(ANNOTATIONS_DIR, "val_split.json"), "w", encoding="utf-8") as f:
        json.dump(filter_coco(coco, val_ids), f, indent=4)

    print(f"Train: {len(train_ids)} images")
    print(f"Val: {len(val_ids)} images")


if __name__ == "__main__":
    run_split()