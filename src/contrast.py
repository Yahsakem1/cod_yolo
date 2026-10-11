import numpy as np
import cv2
from typing import Tuple

def _load_pair(image_path: str, mask_path: str, rgb: bool = False):
    flag = cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(image_path, flag)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise FileNotFoundError(f"Не удалось прочитать {image_path} или {mask_path}")
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return img, mask > 0

def calculate_contrast(image_path: str, mask_path: str) -> float:
    """Контраст между объектом и фоном по яркости: |mean(obj) - mean(bg)|."""
    img, m = _load_pair(image_path, mask_path)
    obj, bg = img[m], img[~m]
    if obj.size == 0 or bg.size == 0:
        return 0.0
    return float(abs(np.mean(obj) - np.mean(bg)))

def calculate_contrast_rgb(image_path: str, mask_path: str) -> float:
    """Евклидово расстояние между средними цветами объекта и фона."""
    img, m = _load_pair(image_path, mask_path, rgb=True)
    obj, bg = img[m], img[~m]
    if obj.size == 0 or bg.size == 0:
        return 0.0
    return float(np.linalg.norm(np.mean(obj, axis=0) - np.mean(bg, axis=0)))

def calculate_contrast_hsv(image_path: str, mask_path: str) -> Tuple[float, float, float]:
    """Разница средних H, S, V. OpenCV: H∈[0,179], S,V∈[0,255]."""
    img, m = _load_pair(image_path, mask_path, rgb=True)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    obj, bg = hsv[m], hsv[~m]
    if obj.size == 0 or bg.size == 0:
        return 0.0, 0.0, 0.0
    h1, s1, v1 = obj.mean(axis=0)
    h2, s2, v2 = bg.mean(axis=0)
    h_diff = abs(h1 - h2)
    h_diff = min(h_diff, 180.0 - h_diff)   # учитываем цикличность тона
    return float(h_diff), float(abs(s1 - s2)), float(abs(v1 - v2))

def calculate_contrast_lab(image_path: str, mask_path: str) -> Tuple[float, float, float]:
    """Разница средних L, A, B. OpenCV 8-bit: все каналы в [0,255]."""
    img, m = _load_pair(image_path, mask_path, rgb=True)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    obj, bg = lab[m], lab[~m]
    if obj.size == 0 or bg.size == 0:
        return 0.0, 0.0, 0.0
    d = obj.mean(axis=0) - bg.mean(axis=0)
    return float(d[0]), float(d[1]), float(d[2])