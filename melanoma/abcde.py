"""Segmentación de la lesión y criterios ABCDE con visión clásica (OpenCV).

Todas las funciones esperan una imagen RGB uint8 de 224x224 (o similar).
"""
import cv2
import numpy as np

# Umbrales por defecto (percentiles sobre ~1.200 imágenes de HAM10000).
# Si existe models/abcde_thr.joblib se usan esos en su lugar.
DEFAULT_THRESHOLDS = {"TH_A": 0.21, "TH_COMP": 1.72, "TH_SOL": 0.90, "TH_CV": 8.3}


def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.inpaint(img, (blackhat > 12).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)


def _pick(m, h, w):
    """Elige la componente más grande y cercana al centro; devuelve máscara 0/1 o None."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(m)
    best, best_score = None, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.01 * h * w:
            continue
        d = np.hypot(cent[i][0] - w / 2, cent[i][1] - h / 2) / (w / 2)
        score = area * (1 - min(d, 1))
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None
    mask = (lbl == best).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, cnts, -1, 1, -1)
    mask = (cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0) > 0.5).astype(np.uint8)
    return mask if mask.sum() > 0 else None


def seg_lab(img):
    """Distancia de color a la piel (estimada en el borde de la imagen)."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    valid = lab[..., 0] > 45  # ignora el viñeteado negro de la dermatoscopia
    ring = np.zeros((h, w), bool)
    ring[:16] = ring[-16:] = ring[:, :16] = ring[:, -16:] = True
    ref = lab[ring & valid]
    ref = np.median(ref, axis=0) if len(ref) > 50 else np.median(lab[ring], axis=0)
    dist = np.linalg.norm(lab - ref, axis=2)
    dist[~valid] = 0
    d8 = np.clip(dist / max(dist.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _pick(m, h, w)


def seg_otsu(img):
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (7, 7), 0)
    gray[gray < 45] = 255
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _pick(m, h, w)


def seg_grab(img):
    h, w = img.shape[:2]
    gc = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    rect = (int(w * .15), int(h * .15), int(w * .7), int(h * .7))
    try:
        cv2.grabCut(np.ascontiguousarray(img), gc, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    return _pick(((gc == 1) | (gc == 3)).astype(np.uint8) * 255, h, w)


def seg_dark(img):
    """Respaldo: mancha más oscura cerca del centro (sirve para lesiones chicas)."""
    h, w = img.shape[:2]
    L = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0], (5, 5), 0)
    yy, xx = np.mgrid[:h, :w]
    center = np.hypot(xx - w / 2, yy - h / 2) < 0.30 * w
    t, _ = cv2.threshold(L[center].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((L < t) & center).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(m)
    best, best_d = None, 1e9
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.003 * h * w:
            continue
        d = np.hypot(cent[i][0] - w / 2, cent[i][1] - h / 2)
        if d < best_d:
            best, best_d = i, d
    if best is None:
        return None
    mask = (lbl == best).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, cnts, -1, 1, -1)
    return mask


def strict_ok(mask):
    h, w = mask.shape
    edge = mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum()
    return mask.mean() < 0.55 and edge < 0.06 * (h + w)


def dark_ok(img, mask):
    """La lesión debe ser más oscura que la piel que la rodea."""
    L = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) - mask
    ring = ring.astype(bool) & (L > 45)
    if ring.sum() < 100:
        return True
    return L[mask > 0].mean() < L[ring].mean() - 8


def segment(img):
    """Prueba cuatro métodos en cadena. Devuelve (máscara, confiable)."""
    first = None
    for fn in (seg_lab, seg_otsu, seg_grab, seg_dark):
        mask = fn(img)
        if mask is not None:
            if strict_ok(mask) and dark_ok(img, mask):
                return mask, True
            if first is None:
                first = mask
    return first, False


def asymmetry(mask):
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    angle = np.degrees(np.arctan2(vt[0][1], vt[0][0]))
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), angle, 1.0)
    M[:, 2] += (w / 2 - c[0], h / 2 - c[1])
    r = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return float(np.mean([1 - (r & f).sum() / (r | f).sum() for f in (r[:, ::-1], r[::-1, :])]))


def border(mask):
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    area, per = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
    return per ** 2 / (4 * np.pi * area), area / cv2.contourArea(cv2.convexHull(cnt)), cnt


def color_var(img, mask):
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if inner.sum() < 200:
        inner = mask
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[inner > 0].astype(np.float32)
    return float(lab.std(axis=0).mean())


def features(img0):
    img = remove_hair(img0)
    mask, ok = segment(img)
    if mask is None:
        return None
    comp, sol, cnt = border(mask)
    return dict(A=asymmetry(mask), comp=comp, sol=sol, Cv=color_var(img, mask),
                D=max(cv2.minAreaRect(cnt)[1]) / img.shape[1] * 100, ok=ok, cnt=cnt)


def evaluar_abcde(img_rgb, th=None):
    """Devuelve un diccionario serializable a JSON con los criterios ABCDE."""
    th = {**DEFAULT_THRESHOLDS, **(th or {})}
    f = features(np.ascontiguousarray(img_rgb))
    base_e = "no evaluable con una sola imagen"
    if f is None or not f["ok"]:
        return {"evaluable": False, "motivo": "segmentación no confiable", "E": base_e, "contorno": None}
    return {
        "evaluable": True,
        "A": {"cumple": bool(f["A"] > th["TH_A"]), "asimetria": round(float(f["A"]), 3)},
        "B": {"cumple": bool(f["comp"] > th["TH_COMP"] or f["sol"] < th["TH_SOL"]),
              "irregularidad": round(float(f["comp"]), 3), "solidez": round(float(f["sol"]), 3)},
        "C": {"cumple": bool(f["Cv"] > th["TH_CV"]), "variabilidad_color": round(float(f["Cv"]), 2)},
        "D": {"porcentaje_ancho_imagen": round(float(f["D"]), 1),
              "nota": "sin escala de referencia no se puede convertir a milímetros"},
        "E": base_e,
        "contorno": f["cnt"].reshape(-1, 2).tolist(),
    }
