"""Clasificador (EfficientNetB0) + calibración + análisis ABCDE en una sola clase."""
import os

import cv2
import joblib
import numpy as np
from PIL import Image

from .abcde import evaluar_abcde

IMG = 224
AVISO = ("Herramienta experimental con fines educativos: no es un diagnóstico médico. "
         "Ante un lunar que cambia, sangra o te preocupa, consultá a un dermatólogo.")


def preparar(imagen):
    """Ruta, PIL.Image o array RGB -> array uint8 (224, 224, 3).

    Usa el mismo reescalado de PIL que en el entrenamiento.
    """
    if isinstance(imagen, (str, os.PathLike)):
        imagen = Image.open(imagen)
    if isinstance(imagen, np.ndarray):
        imagen = Image.fromarray(imagen.astype(np.uint8))
    return np.asarray(imagen.convert("RGB").resize((IMG, IMG)), dtype=np.uint8)


def dibujar_contorno(img224, contorno, color=(0, 255, 0)):
    vis = img224.copy()
    if contorno:
        pts = np.array(contorno, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(vis, [pts], -1, color, 2)
    return vis


def _to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class MelanomaAnalyzer:
    def __init__(self, models_dir="models"):
        import tensorflow as tf  # import diferido: abcde.py se puede usar sin TensorFlow

        self.model = tf.keras.models.load_model(os.path.join(models_dir, "melanoma_model.keras"))
        cal = joblib.load(os.path.join(models_dir, "melanoma_calib.joblib"))
        self.calib, self.thr = cal["calib"], float(cal["thr"])
        p = os.path.join(models_dir, "abcde_thr.joblib")
        self.th = joblib.load(p) if os.path.exists(p) else None

    def riesgo(self, img224):
        p = float(self.model.predict(img224[None].astype("float32"), verbose=0).ravel()[0])
        return float(self.calib.predict_proba(_to_logit(np.array([p])))[:, 1][0])

    def analizar(self, imagen):
        """Devuelve un diccionario serializable a JSON."""
        img = preparar(imagen)
        riesgo = self.riesgo(img)
        return {
            "riesgo_pct": round(riesgo * 100, 1),
            "umbral_alerta_pct": round(self.thr * 100, 1),
            "supera_umbral": bool(riesgo >= self.thr),
            "abcde": evaluar_abcde(img, self.th),
            "aviso": AVISO,
        }
