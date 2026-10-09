"""Uso:  python predict.py ruta/a/imagen.jpg [--salida resultado.png] [--modelos models]"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from PIL import Image  # noqa: E402

from melanoma.predictor import MelanomaAnalyzer, dibujar_contorno, preparar  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Análisis experimental de lesiones (HAM10000).")
    ap.add_argument("imagen")
    ap.add_argument("--modelos", default="models")
    ap.add_argument("--salida", help="guarda la imagen con el contorno de la lesión")
    args = ap.parse_args()

    analizador = MelanomaAnalyzer(args.modelos)
    res = analizador.analizar(args.imagen)

    if args.salida:
        vis = dibujar_contorno(preparar(args.imagen), res["abcde"].get("contorno"))
        Image.fromarray(vis).save(args.salida)

    resumen = {**res, "abcde": {k: v for k, v in res["abcde"].items() if k != "contorno"}}
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
