"""
routes.py
=========
Define os endpoints da API de detecção de fraudes.

Endpoints
---------
GET  /health          : verifica se o serviço está no ar
POST /predict         : avalia uma única transação
POST /predict/batch   : avalia um lote de transações
"""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.predictor import FraudPredictor
from .schemas import (
    TransacaoInput,
    LoteInput,
    PredicaoOutput,
    LoteOutput,
    ItemLoteOutput,
    HealthOutput,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Instância única do predictor — carregada uma vez ao iniciar a API
_predictor: FraudPredictor | None = None


def get_predictor() -> FraudPredictor:
    """Retorna a instância singleton do FraudPredictor."""
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor


def _transacao_para_dataframe(transacao: TransacaoInput) -> pd.DataFrame:
    """Converte um TransacaoInput em DataFrame de uma linha."""
    return pd.DataFrame([transacao.model_dump()])


def _lote_para_dataframe(lote: LoteInput) -> pd.DataFrame:
    """Converte um LoteInput em DataFrame com N linhas."""
    return pd.DataFrame([t.model_dump() for t in lote.transacoes])


# ══════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════════

@router.get(
    "/health",
    response_model=HealthOutput,
    summary="Verifica o status do serviço",
    tags=["Monitoramento"],
)
def health_check() -> HealthOutput:
    """Retorna o status do serviço e metadados do modelo carregado.

    Use este endpoint para confirmar que a API está no ar e o pipeline
    foi carregado corretamente antes de enviar predições.
    """
    try:
        predictor = get_predictor()
        return HealthOutput(
            status="ok",
            modelo=predictor.model_type_,
            threshold=round(predictor.threshold_, 4),
            versao_api="1.0.0",
        )
    except Exception as e:
        logger.exception("Falha no health check")
        raise HTTPException(status_code=503, detail=f"Serviço indisponível: {e}")


# ══════════════════════════════════════════════════════════════════
#  PREDIÇÃO — transação única
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/predict",
    response_model=PredicaoOutput,
    summary="Avalia uma única transação",
    tags=["Predição"],
)
def predict(transacao: TransacaoInput) -> PredicaoOutput:
    """Recebe os dados de uma transação e retorna se é fraude ou não.

    - **fraude**: `1` se fraude detectada, `0` caso contrário
    - **probabilidade**: score de confiança do modelo (0 a 1)
    """
    try:
        predictor = get_predictor()
        df = _transacao_para_dataframe(transacao)

        fraude      = int(predictor.predict(df)[0])
        probabilidade = round(float(predictor.predict_proba(df)[0]), 4)

        logger.info(
            "Predição — fraude=%d | proba=%.4f", fraude, probabilidade
        )

        return PredicaoOutput(fraude=fraude, probabilidade=probabilidade)

    except Exception as e:
        logger.exception("Erro na predição")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")


# ══════════════════════════════════════════════════════════════════
#  PREDIÇÃO — lote
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/predict/batch",
    response_model=LoteOutput,
    summary="Avalia um lote de transações",
    tags=["Predição"],
)
def predict_batch(lote: LoteInput) -> LoteOutput:
    """Recebe uma lista de transações e retorna a predição para cada uma.

    Mais eficiente do que chamar `/predict` múltiplas vezes, pois
    executa o pipeline uma única vez para todas as transações.

    - **total**: número de transações recebidas
    - **total_fraudes**: quantas foram classificadas como fraude
    - **taxa_fraude**: percentual de fraudes no lote
    - **predicoes**: lista com o resultado individual de cada transação
    """
    try:
        predictor = get_predictor()
        df = _lote_para_dataframe(lote)

        fraudes     = predictor.predict(df)
        probabilidades = predictor.predict_proba(df)

        predicoes = [
            ItemLoteOutput(
                indice=i,
                fraude=int(fraudes[i]),
                probabilidade=round(float(probabilidades[i]), 4),
            )
            for i in range(len(fraudes))
        ]

        total_fraudes = int(fraudes.sum())
        taxa_fraude   = round(total_fraudes / len(fraudes), 4)

        logger.info(
            "Predição em lote — %d transações | %d fraudes (%.1f%%)",
            len(fraudes), total_fraudes, taxa_fraude * 100,
        )

        return LoteOutput(
            total=len(fraudes),
            total_fraudes=total_fraudes,
            taxa_fraude=taxa_fraude,
            predicoes=predicoes,
        )

    except Exception as e:
        logger.exception("Erro na predição em lote")
        raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {e}")