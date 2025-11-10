from fastapi import FastAPI
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

@app.get("/")
def root():
    return {"message": "API funcionando 🚀"}

# =========================
# 2. Cargar modelos
# =========================
modelo_causa = joblib.load("modelos/modelo_causa.pkl")
modelo_asiste = joblib.load("modelos/modelo_asiste.pkl")
modelo_nivel = joblib.load("modelos/modelo_nivel_reducido_optimizado.pkl")

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
    IdentidaddeAcuerdoconCostumbres: str = Field(..., example="NO")
    IdentidaddeGenero: str = Field(..., example="Masculino")
    OrientacionSexual: str = Field(..., example="Heterosexual")
    HaEstadoProcesosdeRehabilitacion: str = Field(..., example="NO")
    AsisteaRehabilitacion: str = Field(..., example="NO")
    SuMunicipioTieneServiciodeRehabilitacion: str = Field(..., example="NO")
    UtilizaProductosApoyo: str = Field(..., example="NO")
    LeeyEscribe: str = Field(..., example="SI")
    Trabaja: str = Field(..., example="NO")
    FuenteIngresos: str = Field(..., example="Ninguno")
    IngresoMensualPromedio: str = Field(..., example="0")
    PerteneceaOrganizacionMovimiento: str = Field(..., example="NO")
    TomadeDecisiones: str = Field(..., example="NO")
    RequiereAyudadeOtraPersona: str = Field(..., example="NO")
    UstedVive: str = Field(..., example="Con familia")
    BarrerasFisicas: str = Field(..., example="NO")

# =========================
# 4. Funciones de predicción
# =========================
def convertir_a_df(data: InputData) -> pd.DataFrame:
    """Convierte la entrada JSON en DataFrame para el modelo"""
    return pd.DataFrame([data.dict()])

@app.post("/predict/causa")
def predict_causa(data: InputData):
    df = convertir_a_df(data)
    pred = modelo_causa.predict(df)[0]
    proba = modelo_causa.predict_proba(df).max()
    return {"prediccion": str(pred), "probabilidad": float(proba)}

@app.post("/predict/asiste")
def predict_asiste(data: InputData):
    df = convertir_a_df(data)
    pred = modelo_asiste.predict(df)[0]
    proba = modelo_asiste.predict_proba(df).max()
    return {"prediccion": str(pred), "probabilidad": float(proba)}

@app.post("/predict/nivel")
def predict_nivel(data: InputData):
    df = convertir_a_df(data)
    pred = modelo_nivel.predict(df)[0]
    proba = modelo_nivel.predict_proba(df).max()
    return {"prediccion": str(pred), "probabilidad": float(proba)}
