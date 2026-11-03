"""Servicio HTTP (FastAPI) para el análisis experimental de lesiones.

Este archivo va en la RAÍZ del repositorio y expone la variable `app` (tic ejecuta `uvicorn main:app`).

Local (desde la carpeta del proyecto, con el entorno activado):
    export ALLOW_NO_KEY=1          # solo para pruebas locales, sin clave
    python main.py                 # o: uvicorn main:app --host 127.0.0.1 --port 8000

En tic la clave se carga como variable de entorno MELANOMA_API_KEY desde el panel
(Configuración), ANTES del primer push: sin ella el servicio no arranca.

Variables de entorno opcionales:
    MODELS_DIR, MODEL_VERSION, MAX_IMAGE_MB (10), MAX_PIXELS (40000000),
    DISABLE_DOCS=1 para ocultar /docs y /openapi.json.
"""
import base64
import io
import os
import secrets
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# El filesystem del container es de solo lectura salvo /tmp y data/: Keras escribe su config ahí.
os.environ.setdefault("KERAS_HOME", os.path.join(tempfile.gettempdir(), ".keras"))

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import Depends, FastAPI, File, Header, Request, UploadFile  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from PIL import Image  # noqa: E402

from melanoma.predictor import MelanomaAnalyzer, dibujar_contorno, preparar  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent  # relativo al código, no al directorio desde donde se corre
MODELS_DIR = os.getenv("MODELS_DIR", str(BASE_DIR / "models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "efficientnetb0-ham10000-v1")
API_KEY = os.getenv("MELANOMA_API_key")   # obligatoria: se exige el header X-API-Key
ALLOW_NO_KEY = os.getenv("ALLOW_NO_KEY") == "1"  # solo para pruebas locales
MAX_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
MAX_BYTES = MAX_MB * 1024 * 1024
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "40000000"))  # ancho x alto
ALLOWED = {"image/jpeg", "image/png"}

state = {}
lock = threading.Lock()  # una predicción a la vez: evita problemas de concurrencia con TensorFlow


class ApiError(Exception):
    def __init__(self, status, codigo, mensaje):
        self.status, self.codigo, self.mensaje = status, codigo, mensaje


@asynccontextmanager
async def lifespan(app):
    if not API_KEY and not ALLOW_NO_KEY:
        raise RuntimeError("Definí MELANOMA_API_KEY (o ALLOW_NO_KEY=1 solo para pruebas locales).")
    state["analizador"] = MelanomaAnalyzer(MODELS_DIR)  # el modelo se carga una sola vez
    yield
    state.clear()


_docs = os.getenv("DISABLE_DOCS") != "1"
app = FastAPI(
    title="Análisis experimental de lesiones",
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if _docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if _docs else None,
)


def _error(status, codigo, mensaje):
    return JSONResponse(status_code=status, content={"error": {"codigo": codigo, "mensaje": mensaje}})


def _clave_ok(valor):
    if not API_KEY:  # solo ocurre con ALLOW_NO_KEY=1
        return True
    return secrets.compare_digest((valor or "").encode(), API_KEY.encode())


@app.middleware("http")
async def proteger_analizar(request: Request, call_next):
    """Corta los pedidos no autorizados o demasiado grandes ANTES de que FastAPI procese el cuerpo."""
    if request.method == "POST" and request.url.path == "/analizar":
        if not _clave_ok(request.headers.get("x-api-key")):
            return _error(401, "no_autorizado", "Falta o es incorrecta la clave en el header X-API-Key.")
        largo = request.headers.get("content-length", "")
        if largo.isdigit() and int(largo) > MAX_BYTES + 64 * 1024:
            return _error(413, "imagen_muy_grande", f"La imagen supera los {MAX_MB} MB.")
    return await call_next(request)


@app.exception_handler(ApiError)
async def _handle_api_error(request, exc):
    return _error(exc.status, exc.codigo, exc.mensaje)


@app.exception_handler(RequestValidationError)
async def _handle_validation(request, exc):
    return _error(422, "solicitud_invalida", "Falta el campo 'imagen' (multipart/form-data).")


@app.exception_handler(Exception)
async def _handle_unexpected(request, exc):
    return _error(500, "error_interno", "Error inesperado al analizar la imagen.")


def verificar_clave(x_api_key: str | None = Header(default=None)):
    # Segunda barrera (la primera es el middleware); también hace que /docs ofrezca el campo de la clave.
    if not _clave_ok(x_api_key):
        raise ApiError(401, "no_autorizado", "Falta o es incorrecta la clave en el header X-API-Key.")


@app.get("/health")
def health():
    """Único endpoint que exige tic: tiene que contestar 200 OK, barato y sin clave."""
    return {"ok": True}


@app.get("/salud")
def salud():
    return {"estado": "ok", "version_modelo": MODEL_VERSION}


@app.post("/analizar", dependencies=[Depends(verificar_clave)])
def analizar(imagen: UploadFile = File(...)):
    if imagen.content_type not in ALLOWED:
        raise ApiError(415, "formato_no_soportado", "Solo se aceptan imágenes JPEG o PNG.")

    datos = imagen.file.read(MAX_BYTES + 1)
    if len(datos) > MAX_BYTES:
        raise ApiError(413, "imagen_muy_grande", f"La imagen supera los {MAX_MB} MB.")

    try:
        pil = Image.open(io.BytesIO(datos))  # lectura perezosa: solo lee el encabezado
    except Exception:
        raise ApiError(400, "imagen_invalida", "No se pudo leer la imagen.")
    if pil.width * pil.height > MAX_PIXELS:
        raise ApiError(413, "imagen_muy_grande", "Las dimensiones de la imagen son demasiado grandes.")
    try:
        pil.load()
    except Exception:
        raise ApiError(400, "imagen_invalida", "No se pudo leer la imagen.")

    with lock:
        resultado = state["analizador"].analizar(pil)

    vis = dibujar_contorno(preparar(pil), resultado["abcde"].get("contorno"))
    buf = io.BytesIO()
    Image.fromarray(vis).save(buf, format="PNG")

    resultado["version_modelo"] = MODEL_VERSION
    resultado["imagen_con_contorno"] = base64.b64encode(buf.getvalue()).decode("ascii")
    return resultado


if __name__ == "__main__":  # solo para correr en local; tic no lo usa
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)