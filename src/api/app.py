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

import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# ── Timeout Middleware ─────────────────────────────────────────────────────────
# Rejeita requisições que demorem mais de 10 segundos com erro 503.
# Evita que o servidor fique enfileirando requisições indefinidamente
# sob alta carga, retornando um erro claro ao invés de travar.
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=10.0)
    except asyncio.TimeoutError:
        logging.warning(
            "Timeout de 10s atingido para %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "Request timeout — servidor sobrecarregado"},
        )

# ── CORS ───────────────────────────────────────────────────────────────────────
# Permite chamadas de qualquer origem em desenvolvimento.
# Em produção, substitua ["*"] pela lista de domínios permitidos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rotas ──────────────────────────────────────────────────────────────────────
app.include_router(router)