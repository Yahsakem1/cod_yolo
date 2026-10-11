"""
Утилиты для работы с COCO-аннотациями: геометрия (площади, IoU),
фильтрация вырожденных полигонов и проверки целостности/качества
датасета (используются в EDA и как sanity-check перед train/val/test
split и конвертацией в YOLO).
"""

import json
import os
from collections import Counter


# ==========================
# Базовые геометрические функции
# ==========================
def _load(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def polygon_area(coords):
    """Площадь полигона (shoelace). coords = [x1,y1,x2,y2,...]."""
    n = len(coords) // 2
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = coords[2 * i], coords[2 * i + 1]
        x2, y2 = coords[2 * ((i + 1) % n)], coords[2 * ((i + 1) % n) + 1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def bbox_area(bbox):
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


def bbox_from_all_polygons(segmentation):
    """Восстанавливает bbox [x, y, w, h] по всем точкам всех полигонов аннотации."""
    all_xs, all_ys = [], []
    for poly in segmentation:
        all_xs.extend(poly[0::2])
        all_ys.extend(poly[1::2])
    return [min(all_xs), min(all_ys), max(all_xs) - min(all_xs), max(all_ys) - min(all_ys)]


def iou(box1, box2):
    """IoU двух bbox в формате [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union else 0


def filter_degenerate_polygons(segmentation, eps=1e-6):
    """
    Убирает вырожденные подполигоны (area <= eps) — артефакт трассировки
    контуров на тонких выступах маски (усики, лапки, тонкие полосы и т.п.).
    Возвращает (kept_polygons, removed_count).
    """
    kept, removed = [], 0
    for poly in segmentation:
        if len(poly) < 6 or len(poly) % 2 != 0:
            removed += 1
            continue
        if polygon_area(poly) <= eps:
            removed += 1
            continue
        kept.append(poly)
    return kept, removed


def _fmt_counter(counter, top=None, total=None):
    items = counter.most_common(top) if top else counter.most_common()
    if total is None:
        total = sum(counter.values()) or 1
    return "\n".join(
        f"    {k}: {v} ({100.0 * v / total:.2f}%)" for k, v in items
    )


# ==========================
# 1. Общая сводка
# ==========================
def eda_summary(json_path, split_name=None):
    """Общая сводка по датасету: число изображений/аннотаций/категорий."""
    split_name = split_name or os.path.basename(json_path)
    coco = _load(json_path)

    images      = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories  = coco.get("categories", [])

    print(f"\n{'=' * 60}")
    print(f"  Сводка: {split_name}")
    print(f"{'=' * 60}")
    print(f"  Изображений:  {len(images)}")
    print(f"  Аннотаций:    {len(annotations)}")
    print(f"  Категорий:    {len(categories)}")

    if images:
        ws = [im["width"]  for im in images]
        hs = [im["height"] for im in images]
        print(f"  Размеры изображений:")
        print(f"    W: min={min(ws)} max={max(ws)} avg={sum(ws)/len(ws):.1f}")
        print(f"    H: min={min(hs)} max={max(hs)} avg={sum(hs)/len(hs):.1f}")

    print(f"  Категории:")
    for c in categories:
        print(f"    id={c.get('id')}  name={c.get('name')}")

    if annotations:
        avg = len(annotations) / len(images) if images else 0
        print(f"  Объектов на изображение (avg): {avg:.2f}")

    return coco


# ==========================
# 2. Распределение по классам
# ==========================
def eda_class_distribution(coco, split_name="train", top=None):
    """Сколько объектов в каждом классе + доля."""
    print(f"\n--- Распределение по классам: {split_name} ---")

    id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
    counter = Counter(a["category_id"] for a in coco.get("annotations", []))
    total = sum(counter.values()) or 1

    if not counter:
        print("    (нет аннотаций)")
        return counter

    rows = sorted(counter.items(), key=lambda kv: -kv[1])
    if top:
        rows = rows[:top]
    for cid, cnt in rows:
        print(f"    {id_to_name.get(cid, cid)} (id={cid}): {cnt} ({100.0*cnt/total:.2f}%)")

    # дисбаланс
    vals = list(counter.values())
    if len(vals) > 1:
        print(f"    Дисбаланс: max/min = {max(vals)/min(vals):.1f}")
    return counter


# ==========================
# 3. Площади bbox
# ==========================
def eda_bbox_areas(coco, split_name="train"):
    """Статистика площадей bbox + количество нулевых/мелких."""
    print(f"\n--- Площади bbox: {split_name} ---")

    anns = coco.get("annotations", [])
    if not anns:
        print("    (нет аннотаций)")
        return

    areas = []
    zero, neg, tiny, small, medium, large = 0, 0, 0, 0, 0, 0
    per_class_zero = Counter()

    for a in anns:
        b = a.get("bbox", [0, 0, 0, 0])
        w, h = b[2], b[3]
        if w < 0 or h < 0:
            neg += 1
            continue
        ar = w * h
        areas.append(ar)
        if ar == 0:
            zero += 1
            per_class_zero[a["category_id"]] += 1
        elif ar < 1:
            tiny += 1
        elif ar < 32 * 32:
            small += 1
        elif ar < 96 * 96:
            medium += 1
        else:
            large += 1

    if not areas:
        print("    (нет валидных bbox)")
        return

    areas_sorted = sorted(areas)
    n = len(areas_sorted)
    def pct(p): return areas_sorted[min(n - 1, int(p * n))]
    print(f"    Всего bbox:      {n}")
    print(f"    min / median / max: {areas_sorted[0]:.1f} / {pct(0.5):.1f} / {areas_sorted[-1]:.1f}")
    print(f"    mean:            {sum(areas)/n:.1f}")
    print(f"    p90 / p99:       {pct(0.9):.1f} / {pct(0.99):.1f}")
    print(f"    Площадь == 0:    {zero}")
    print(f"    Площадь < 1:     {tiny}")
    print(f"    Площадь < 32²:   {small}  (COCO 'small')")
    print(f"    Площадь < 96²:   {medium} (COCO 'medium')")
    print(f"    Площадь >= 96²:  {large}  (COCO 'large')")
    if neg:
        print(f"    [!] Отрицательные w/h: {neg}")
    if zero:
        id_to_name = {c["id"]: c.get("name", c["id"]) for c in coco.get("categories", [])}
        print(f"    Нулевые bbox по классам:")
        for cid, cnt in per_class_zero.most_common():
            print(f"      {id_to_name.get(cid, cid)}: {cnt}")


# ==========================
# 4. Площади полигонов (segmentation)
# ==========================
def eda_polygon_areas(coco, split_name="train"):
    """Проверка segmentation: пустые, невалидные, нулевой площади."""
    print(f"\n--- Площади полигонов: {split_name} ---")

    anns = coco.get("annotations", [])
    if not anns:
        print("    (нет аннотаций)")
        return

    empty_seg = 0
    invalid_poly = 0
    zero_poly = 0
    multi_poly = 0
    rle_seg = 0
    total_polys = 0
    poly_areas = []

    for a in anns:
        seg = a.get("segmentation", None)
        if not seg:
            empty_seg += 1
            continue
        if isinstance(seg, dict):
            rle_seg += 1
            continue
        if not isinstance(seg, list):
            invalid_poly += 1
            continue
        if len(seg) > 1:
            multi_poly += 1
        for poly in seg:
            total_polys += 1
            if len(poly) < 6 or len(poly) % 2 != 0:
                invalid_poly += 1
                continue
            pa = polygon_area(poly)
            poly_areas.append(pa)
            if pa <= 1e-6:
                zero_poly += 1

    print(f"    Всего аннотаций:         {len(anns)}")
    print(f"    Всего полигонов:         {total_polys}")
    print(f"    Мультиполигональных:     {multi_poly}")
    print(f"    Пустая segmentation:     {empty_seg}")
    print(f"    RLE segmentation:        {rle_seg}")
    print(f"    Невалидных полигонов:    {invalid_poly}")
    print(f"    Нулевой площади (==0):   {zero_poly}")

    if poly_areas:
        pa_sorted = sorted(poly_areas)
        n = len(pa_sorted)
        print(f"    Площадь полигонов:")
        print(f"      min / median / max: {pa_sorted[0]:.2f} / {pa_sorted[n//2]:.2f} / {pa_sorted[-1]:.2f}")
        print(f"      mean:               {sum(pa_sorted)/n:.2f}")


# ==========================
# 5. Согласованность bbox и segmentation
# ==========================
def eda_bbox_vs_seg(coco, split_name="train", iou_thr=0.5):
    """Сравнивает ann['bbox'] с bbox, восстановленным из полигонов."""
    print(f"\n--- bbox vs segmentation (IoU<{iou_thr}): {split_name} ---")

    anns = coco.get("annotations", [])
    if not anns:
        print("    (нет аннотаций)")
        return

    bad = []
    checked = 0
    for a in anns:
        seg = a.get("segmentation")
        if not seg or not isinstance(seg, list):
            continue
        xs, ys = [], []
        ok = True
        for poly in seg:
            if len(poly) < 2 or len(poly) % 2 != 0:
                ok = False
                break
            xs.extend(poly[0::2])
            ys.extend(poly[1::2])
        if not ok or not xs:
            continue
        bx = min(xs); by = min(ys)
        bw = max(xs) - bx; bh = max(ys) - by
        bbox_seg = [bx, by, bw, bh]
        bbox_ann = a.get("bbox", [0, 0, 0, 0])

        # IoU
        xa1, ya1, xa2, ya2 = bbox_ann[0], bbox_ann[1], bbox_ann[0]+bbox_ann[2], bbox_ann[1]+bbox_ann[3]
        xb1, yb1, xb2, yb2 = bbox_seg[0], bbox_seg[1], bbox_seg[0]+bbox_seg[2], bbox_seg[1]+bbox_seg[3]
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw * ih
        union = bbox_ann[2]*bbox_ann[3] + bbox_seg[2]*bbox_seg[3] - inter
        iou_v = inter / union if union > 0 else 0
        checked += 1
        if iou_v < iou_thr:
            bad.append((a["id"], round(iou_v, 3), bbox_ann, bbox_seg))

    print(f"    Проверено аннотаций:  {checked}")
    print(f"    Расхождений (IoU<{iou_thr}): {len(bad)}")
    for row in bad[:10]:
        print(f"      ann_id={row[0]}  IoU={row[1]}  bbox={row[2]}  seg_bbox={row[3]}")
    if len(bad) > 10:
        print(f"      ... ещё {len(bad) - 10}")


# ==========================
# 6. Выход за границы изображения
# ==========================
def eda_out_of_bounds(coco, split_name="train"):
    """Проверяет, выходят ли bbox и полигоны за пределы изображения."""
    print(f"\n--- Выход за границы: {split_name} ---")

    id_to_img = {im["id"]: im for im in coco.get("images", [])}
    anns = coco.get("annotations", [])
    if not anns:
        print("    (нет аннотаций)")
        return

    bbox_oob = 0
    poly_oob = 0
    for a in anns:
        img = id_to_img.get(a["image_id"])
        if not img:
            continue
        W, H = img["width"], img["height"]

        b = a.get("bbox", [0, 0, 0, 0])
        if b[0] < -1 or b[1] < -1 or b[0] + b[2] > W + 1 or b[1] + b[3] > H + 1:
            bbox_oob += 1

        seg = a.get("segmentation")
        if seg and isinstance(seg, list):
            xs, ys = [], []
            for poly in seg:
                if len(poly) >= 2 and len(poly) % 2 == 0:
                    xs.extend(poly[0::2]); ys.extend(poly[1::2])
            if xs and (min(xs) < -1 or min(ys) < -1 or max(xs) > W + 1 or max(ys) > H + 1):
                poly_oob += 1

    print(f"    bbox вне границ:      {bbox_oob}")
    print(f"    полигоны вне границ:  {poly_oob}")


# ==========================
# 7. Дубликаты и целостность
# ==========================
def eda_integrity(coco, split_name="train"):
    """Дубликаты id/имён, orphan image_id, unknown category_id."""
    print(f"\n--- Целостность: {split_name} ---")

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    img_ids = [im["id"] for im in images]
    img_names = [im["file_name"] for im in images]
    cat_ids = {c["id"] for c in cats}

    dup_img_ids = [k for k, v in Counter(img_ids).items() if v > 1]
    dup_img_names = [k for k, v in Counter(img_names).items() if v > 1]

    ann_ids = [a.get("id") for a in anns]
    dup_ann_ids = [k for k, v in Counter(ann_ids).items() if v > 1]

    valid_img_ids = set(img_ids)
    orphan = sum(1 for a in anns if a.get("image_id") not in valid_img_ids)
    unknown_cat = sum(1 for a in anns if a.get("category_id") not in cat_ids)

    print(f"    Дубликаты image.id:        {len(dup_img_ids)}")
    print(f"    Дубликаты image.file_name: {len(dup_img_names)}")
    print(f"    Дубликаты annotation.id:   {len(dup_ann_ids)}")
    print(f"    Orphan image_id:           {orphan}")
    print(f"    Unknown category_id:       {unknown_cat}")

    # картинки без аннотаций
    img_with_ann = {a["image_id"] for a in anns}
    no_ann = len(valid_img_ids - img_with_ann)
    print(f"    Изображений без аннотаций: {no_ann}")

    # аннотации с несколькими объектами одного класса на одном изображении
    # (просто как метрика плотности)
    per_img = Counter(a["image_id"] for a in anns)
    if per_img:
        top = per_img.most_common(1)[0]
        print(f"    Макс. аннотаций на изображении: {top[1]} (image_id={top[0]})")
