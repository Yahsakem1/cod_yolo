import numpy as np 
import cv2 

def compute_texture_zones(image_path: str, mask_path: str,
                           ring_scale: float = 0.3,
                           ring_min: int = 5,
                           ring_max: int = 80,
                           local_ksize: int = 5):
    """
    Считает среднюю локальную дисперсию (текстуру) в трёх зонах:
    - object    : сам объект (по маске)
    - local_bg  : кольцо вокруг объекта, ширина масштабируется под размер объекта
    - global_bg : всё, что осталось (дальний фон)

    ring_scale - доля от sqrt(bbox_area) объекта, определяющая ширину кольца
    ring_min/ring_max - защита от слишком узкого/слишком широкого кольца
    local_ksize - размер окна локальной дисперсии текстуры
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None
    if img.shape != mask.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    object_mask = mask > 0
    if object_mask.sum() == 0:
        return None

    y_idx, x_idx = np.where(object_mask)
    bbox_h = y_idx.max() - y_idx.min() + 1
    bbox_w = x_idx.max() - x_idx.min() + 1

    ring_size = int(np.sqrt(bbox_h * bbox_w) * ring_scale)
    ring_size = int(np.clip(ring_size, ring_min, ring_max))
    if ring_size % 2 == 0:
        ring_size += 1  # нечётный размер ядра для симметричной дилатации

    obj_uint8 = object_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_size, ring_size))
    dilated = cv2.dilate(obj_uint8, kernel, iterations=1).astype(bool)

    local_bg_mask = dilated & ~object_mask
    global_bg_mask = ~dilated

    gray = img.astype(np.float32)
    mean = cv2.blur(gray, (local_ksize, local_ksize))
    mean_sq = cv2.blur(gray ** 2, (local_ksize, local_ksize))
    local_var_map = np.clip(mean_sq - mean ** 2, 0, None)

    def safe_mean(m):
        vals = local_var_map[m]
        return float(np.mean(vals)) if vals.size > 0 else np.nan

    return {
        "object_var": safe_mean(object_mask),
        "local_bg_var": safe_mean(local_bg_mask),
        "global_bg_var": safe_mean(global_bg_mask),
        "ring_size_used": ring_size, 
    }
