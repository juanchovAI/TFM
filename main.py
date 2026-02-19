from fastapi import FastAPI, File, UploadFile, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import traceback
import logging
from fastapi.middleware.cors import CORSMiddleware
import os
import io
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# =========================
# 1. Inicializar FastAPI
# =========================
app = FastAPI(
    title="Predicción de Educación",
    description="API para predecir CausaNoEstudia, AsisteaInstitucionEducativa y NivelEducativo reducido",
    version="1.0"
)

# Allow CORS for any origin by default (adjust in production)
origins = [
    "https://tfm-ui2025.netlify.app",  # Tu dominio de frontend
    "http://localhost:3000",           # Por si pruebas localmente
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],               # Permite POST, GET, etc.
    allow_headers=["*"],               # Permite todos los headers
)
# Configure logger
logger = logging.getLogger("tfm_api")
logger.setLevel(logging.INFO)


# Global exception handler: log full traceback but return a generic error to clients
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    # Log the full traceback for server-side debugging
    logger.error("Unhandled exception: %s", tb)
    # Return a generic message to the client (avoid leaking internal details)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.get("/")
def root():
    return {"message": "API funcionando 🚀"}

# =========================
# 2. Cargar modelos
# =========================
from pathlib import Path

# Resolve model paths relative to this file so loading works regardless of current working directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelos"

try:
    modelo_causa = joblib.load(str(MODEL_DIR / "modelo_causa.pkl"))
    modelo_asiste = joblib.load(str(MODEL_DIR / "modelo_asiste.pkl"))
    modelo_nivel = joblib.load(str(MODEL_DIR / "modelo_nivel_reducido_optimizado.pkl"))
    logger.info("Models loaded from %s", MODEL_DIR)
except Exception as e:
    tb = traceback.format_exc()
    logger.error("Failed to load models from %s: %s\n%s", MODEL_DIR, e, tb)
    # Re-raise so the application fails fast if models are missing/corrupt
    raise

# =========================
# 3. Definir schema de entrada
# =========================
class InputData(BaseModel):
    Grupo_de_Edad: str = Field(..., example="15 a 19")
    Localidad: str = Field(..., example="Rural")
    Cat_Fisica: str = Field(..., example="Ninguna")
    Cat_Visual: str = Field(..., example="Ninguna")
    Cat_Auditiva: str = Field(..., example="Ninguna")
    Cat_Intelectual: str = Field(..., example="Ninguna")
    Cat_Psicosocial: str = Field(..., example="Ninguna")
    Cat_Sordoceguera: str = Field(..., example="Ninguna")
    Cat_Multiple: str = Field(..., example="Ninguna")
    Congnicion: str = Field(..., example="Sin dificultad")
    Movilidad: str = Field(..., example="Sin dificultad")
    Cuidado_Personal: str = Field(..., example="Sin dificultad")
    Relaciones: str = Field(..., example="Sin dificultad")
    Actividades_vida_diaria: str = Field(..., example="Sin dificultad")
    Global: str = Field(..., example="Sin dificultad")
    CausaDeficiencia: str = Field(..., example="Ninguna")
    IdentidaddeAcuerdoconCostumbres: str = Field(..., json_schema_extra={"example": "NO"})
    IdentidaddeGenero: str = Field(..., example="Masculino")
    OrientacionSexual: str = Field(..., example="Heterosexual")
    HaEstadoProcesosdeRehabilitacion: str = Field(..., json_schema_extra={"example": "NO"})
    AsisteaRehabilitacion: str = Field(..., json_schema_extra={"example": "NO"})
    SuMunicipioTieneServiciodeRehabilitacion: str = Field(..., json_schema_extra={"example": "NO"})
    UtilizaProductosApoyo: str = Field(..., json_schema_extra={"example": "NO"})
    LeeyEscribe: str = Field(..., example="SI")
    Trabaja: str = Field(..., json_schema_extra={"example": "NO"})
    FuenteIngresos: str = Field(..., example="Ninguno")
    IngresoMensualPromedio: str = Field(..., example="0")
    PerteneceaOrganizacionMovimiento: str = Field(..., json_schema_extra={"example": "NO"})
    TomadeDecisiones: str = Field(..., json_schema_extra={"example": "NO"})
    RequiereAyudadeOtraPersona: str = Field(..., json_schema_extra={"example": "NO"})
    UstedVive: str = Field(..., example="Con familia")
    BarrerasFisicas: str = Field(..., json_schema_extra={"example": "NO"})

# =========================
# 4. Funciones de predicción
# =========================
def convertir_a_df(data: InputData) -> pd.DataFrame:
    """Convierte la entrada JSON en DataFrame para el modelo"""
    return pd.DataFrame([data.dict()])


# ----------------------
# API Key dependency
# ----------------------
def api_key_auth(x_api_key: str = Header(None)):
    expected = os.getenv("API_KEY")
    # If no API_KEY set in env, skip auth (backwards compatible)
    if expected is None:
        return True
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True

@app.post("/predict/causa")
def predict_causa(data: InputData, authorized: bool = Depends(api_key_auth)):
    # Debug: log incoming data
    logger.info("/predict/causa called with data: %s", data.dict())
    try:
        df = convertir_a_df(data)
        pred = modelo_causa.predict(df)[0]
        proba = modelo_causa.predict_proba(df).max()
        return {"prediccion": str(pred), "probabilidad": float(proba)}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error in predict_causa: %s\n%s", e, tb)
        # Log error server-side and return a generic message to the client
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.post("/predict/asiste")
def predict_asiste(data: InputData, authorized: bool = Depends(api_key_auth)):
    logger.info("/predict/asiste called with data: %s", data.dict())
    try:
        df = convertir_a_df(data)
        pred = modelo_asiste.predict(df)[0]
        proba = modelo_asiste.predict_proba(df).max()
        return {"prediccion": str(pred), "probabilidad": float(proba)}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error in predict_asiste: %s\n%s", e, tb)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.post("/predict/nivel")
def predict_nivel(data: InputData, authorized: bool = Depends(api_key_auth)):
    logger.info("/predict/nivel called with data: %s", data.dict())
    try:
        df = convertir_a_df(data)
        pred = modelo_nivel.predict(df)[0]
        proba = modelo_nivel.predict_proba(df).max()
        return {"prediccion": str(pred), "probabilidad": float(proba)}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error in predict_nivel: %s\n%s", e, tb)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), authorized: bool = Depends(api_key_auth)):
    """Recibe un archivo Excel o CSV, ejecuta las 3 predicciones por fila y devuelve un Excel con resultados."""
    contents = await file.read()
    # Verificación de tamaño máximo de upload (en MB). Por defecto 20 MB.
    try:
        max_mb = int(os.getenv('MAX_UPLOAD_MB', '20'))
    except Exception:
        max_mb = 20
    max_bytes = max_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Archivo demasiado grande. Tamaño máximo permitido: {max_mb} MB")
    try:
        if file.filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            # assume CSV
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo el archivo: {e}")

    # Ensure expected columns exist by filling missing with empty string
    expected_cols = [field for field in InputData.__fields__.keys()]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""

    results = []
    for _, row in df.iterrows():
        row_dict = {c: (row[c] if pd.notna(row[c]) else "") for c in expected_cols}
        row_df = pd.DataFrame([row_dict])
        # Predictions
        try:
            p_causa = modelo_causa.predict(row_df)[0]
            prob_causa = float(modelo_causa.predict_proba(row_df).max())
        except Exception:
            p_causa = ""
            prob_causa = None
        try:
            p_asiste = modelo_asiste.predict(row_df)[0]
            prob_asiste = float(modelo_asiste.predict_proba(row_df).max())
        except Exception:
            p_asiste = ""
            prob_asiste = None
        try:
            p_nivel = modelo_nivel.predict(row_df)[0]
            prob_nivel = float(modelo_nivel.predict_proba(row_df).max())
        except Exception:
            p_nivel = ""
            prob_nivel = None

        results.append({
            **row_dict,
            'pred_causa': str(p_causa), 'prob_causa': prob_causa,
            'pred_asiste': str(p_asiste), 'prob_asiste': prob_asiste,
            'pred_nivel': str(p_nivel), 'prob_nivel': prob_nivel,
        })

    out_df = pd.DataFrame(results)

    # Write to Excel in-memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        out_df.to_excel(writer, index=False, sheet_name='predictions')
    output.seek(0)

    headers = {
        'Content-Disposition': f'attachment; filename="predictions_{file.filename.rsplit(".",1)[0]}.xlsx"'
    }
    return StreamingResponse(output, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers=headers)
