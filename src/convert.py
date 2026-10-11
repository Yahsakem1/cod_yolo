import json
import os
from collections import defaultdict
from src.coco_utils import polygon_area, iou, filter_degenerate_polygons
# ==========================
# Настройки
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
JSON_TRAIN = os.path.join(DATA_DIR, "annotations", "train_split.json")
JSON_VAL = os.path.join(DATA_DIR, "annotations", "val_split.json")
JSON_TEST = os.path.join(DATA_DIR, "annotations", "CAM_Instance_Test.json")  
# ==========================
# Вспомогательные функции
# ==========================

def bbox_from_all_polygons(segmentation):
    all_xs, all_ys = [], []
    for poly in segmentation:
        all_xs.extend(poly[0::2])
        all_ys.extend(poly[1::2])
    return [min(all_xs), min(all_ys), max(all_xs) - min(all_xs), max(all_ys) - min(all_ys)]
# ==========================
# Конвертер
# ==========================
def coco_to_yolo(json_path: str, split: str) -> None:
    print(f"\nОбработка: {split}")

    label_dir = os.path.join(DATA_DIR, "labels", split)
    os.makedirs(label_dir, exist_ok=True)
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    bad_bbox = []
    multi_poly_count = 0
    empty_seg_count = 0
    degenerate_removed_total = 0
    fully_degenerate_anns = []
    count = 0
    for img_id, anns in anns_by_image.items():
        img = id_to_img[img_id]
        W, H = img["width"], img["height"]
        filename = img["file_name"]
        txt_name = filename.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(label_dir, txt_name)
        lines = []
        for ann in anns:
            segmentation = ann["segmentation"]
            if not segmentation:
                empty_seg_count += 1
                continue

            # фильтруем вырожденные (area ~ 0) подполигоны — артефакт
            # трассировки контуров на тонких выступах маски (см. EDA)
            segmentation, removed = filter_degenerate_polygons(segmentation)
            degenerate_removed_total += removed

            if not segmentation:
                # все подполигоны были вырожденными — аннотация целиком
                # непригодна, логируем отдельно и пропускаем
                fully_degenerate_anns.append((filename, ann["id"]))
                continue

            if len(segmentation) > 1:
                multi_poly_count += 1
            bbox_all = bbox_from_all_polygons(segmentation)
            if iou(ann["bbox"], bbox_all) < 0.5:
                bad_bbox.append((filename, ann["id"]))
            for poly in segmentation:
                coords = []
                for i in range(0, len(poly), 2):
                    x = poly[i] / W
                    y = poly[i + 1] / H
                    coords.extend([round(x, 6), round(y, 6)])
                lines.append("0 " + " ".join(map(str, coords)))
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        count += 1
    print(f"Создано label файлов: {count}")
    print(f"Объектов с несколькими полигонами: {multi_poly_count}")
    print(f"Аннотаций с пустой segmentation: {empty_seg_count}")
    print(f"Вырожденных подполигонов отфильтровано: {degenerate_removed_total}")
    print(f"Аннотаций, где ВСЕ подполигоны вырождены (пропущены целиком): {len(fully_degenerate_anns)}")
    if fully_degenerate_anns:
        print("Примеры:", fully_degenerate_anns[:5])
    print(f"Проблемных bbox (после фикса): {len(bad_bbox)}")
    if bad_bbox:
        print("Примеры:", bad_bbox[:5])
# ==========================
# Запуск
# ==========================
if __name__ == "__main__":
    coco_to_yolo(JSON_TRAIN, "train")
    coco_to_yolo(JSON_VAL, "val")
    if os.path.exists(JSON_TEST):
        coco_to_yolo(JSON_TEST, "test")
    else:
        print(f"\n[!] Test JSON не найден по пути: {JSON_TEST} — пропущено")