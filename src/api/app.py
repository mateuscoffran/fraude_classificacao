"""
app.py
======
Instância principal do FastAPI e ponto de entrada da aplicação.

Para rodar localmente:
    uvicorn src.api.app:app --reload

A documentação interativa fica disponível em:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ── Instância da aplicação ─────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detector API",
    description=(
        "API de detecção de transações fraudulentas.\n\n"
        "Utiliza um pipeline XGBoost treinado com engenharia de features "
        "completa (KNN Imputer, CatBoost Encoding, KMeans Discretizer, "
        "features polinomiais e seleção via Boruta).\n\n"
        "**Endpoints disponíveis:**\n"
        "- `GET /health` — status do serviço\n"
        "- `POST /predict` — avalia uma única transação\n"
        "- `POST /predict/batch` — avalia um lote de transações"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Permite chamadas de qualquer origem em desenvolvimento.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rotas ──────────────────────────────────────────────────────────────────────
app.include_router(router)