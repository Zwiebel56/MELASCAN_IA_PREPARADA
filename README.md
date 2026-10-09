# Detección experimental de melanoma (HAM10000)

Proyecto educativo: clasifica una imagen dermatoscópica como melanoma / no melanoma,
estima un riesgo en porcentaje (calibrado) y calcula de forma aproximada los criterios
ABCDE. **No es un dispositivo médico ni un diagnóstico.**

## Instalación
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```
Copiá los archivos del modelo a `models/` (ver `models/README.md`).

## Uso
```bash
python predict.py ruta/a/imagen.jpg --salida resultado.png
```
Desde Python:
```python
from melanoma.predictor import MelanomaAnalyzer
resultado = MelanomaAnalyzer("models").analizar("imagen.jpg")   # dict serializable a JSON
```

## Resultados (conjunto de test, 1.675 imágenes, split por `lesion_id`)
- AUC 0,837. Con el umbral de alerta elegido en validación (recall ≥ 90 %): recall 87 %,
  falsas alarmas ~42 %, precisión ~21 %.
- Calibración Platt ajustada en validación: el porcentaje es relativo a la prevalencia de HAM10000 (~11 %).

## Limitaciones conocidas
- Entrenado solo con imágenes **dermatoscópicas**; con fotos de celular el resultado no es confiable.
- El modelo depende parcialmente del contexto de la imagen: con la lesión tapada conserva AUC ~0,74.
- ABCDE con visión clásica: depende de la segmentación, que falla en lesiones difusas. D se da como
  porcentaje del ancho de la imagen (sin escala) y E no es evaluable con una sola foto.
- Clasificación binaria: `akiec` y `bcc` cuentan como "no melanoma".

## Datos
HAM10000: Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large collection of
multi-source dermatoscopic images of common pigmented skin lesions.* Sci Data 2018.
Revisá su licencia (CC BY-NC 4.0, uso no comercial) antes de reutilizar o publicar.
