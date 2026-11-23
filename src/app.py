"""API FastAPI para inferência do modelo de churn."""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.schemas import ClienteEntrada, PredicaoSaida

CAMINHO_MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo_mlp.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o pipeline uma única vez na inicialização do serviço."""
    try:
        app.state.modelo = joblib.load(CAMINHO_MODELO)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Modelo não encontrado em {CAMINHO_MODELO}. Execute o notebook de treinamento."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Não foi possível carregar o modelo: {exc}") from exc
    yield


app = FastAPI(
    title="Predição de Churn - AT2",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/info")
def info() -> dict[str, str]:
    return {"status": "ativo", "framework": "FastAPI", "modelo": "MLPClassifier"}


@app.post("/predict", response_model=PredicaoSaida)
def predict(entrada: ClienteEntrada, request: Request) -> PredicaoSaida:
    """Executa uma inferência com dados brutos validados pelo Pydantic."""
    if not hasattr(request.app.state, "modelo"):
        raise HTTPException(status_code=503, detail="Modelo ainda não está disponível.")

    dados = pd.DataFrame([entrada.model_dump()])
    modelo = request.app.state.modelo
    classe = str(modelo.predict(dados)[0])
    classes = list(modelo.classes_)
    indice = classes.index(classe)
    probabilidade = float(modelo.predict_proba(dados)[0][indice])
    return PredicaoSaida(classe_predita=classe, probabilidade=probabilidade)
