"""
schemas.py
==========
Modelos Pydantic que definem o contrato de entrada e saída da API.

Entrada
-------
TransacaoInput      : uma única transação (todos os campos do fraude.xlsx)
LoteInput           : lista de transações

Saída
-----
PredicaoOutput      : resultado para uma transação
LoteOutput          : resultado para um lote
HealthOutput        : resposta do endpoint de saúde
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════
#  ENTRADA
# ══════════════════════════════════════════════════════════════════

class TransacaoInput(BaseModel):
    """Representa uma única transação a ser avaliada pelo modelo.

    Todos os campos correspondem às colunas do fraude.xlsx.
    Campos opcionais refletem colunas que podem ter valores ausentes.
    """

    score_1:              int
    score_2:              float
    score_3:              float
    score_4:              float
    score_5:              float
    score_6:              float
    pais:                 Optional[str] = None
    score_7:              int
    categoria_produto:    str
    score_8:              float
    score_9:              float
    score_10:             float
    entrega_doc_1:        int
    entrega_doc_2:        Optional[str] = Field(
                              default=None,
                              description="Valores aceitos: 'Y', 'N' ou null"
                          )
    entrega_doc_3:        Optional[str] = Field(
                              default=None,
                              description="Valores aceitos: 'Y', 'N' ou null"
                          )
    valor_compra:         float
    score_fraude_modelo:  Optional[int] = None
    produto:              Optional[str] = None
    data_compra:          Optional[datetime] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "score_1": 4,
                "score_2": 0.7685,
                "score_3": 94436.24,
                "score_4": 20.0,
                "score_5": 0.44,
                "score_6": 1.0,
                "pais": "BR",
                "score_7": 5,
                "categoria_produto": "cat_8d714cd",
                "score_8": 0.88,
                "score_9": 240.0,
                "score_10": 102.0,
                "entrega_doc_1": 1,
                "entrega_doc_2": None,
                "entrega_doc_3": "N",
                "valor_compra": 5.64,
                "score_fraude_modelo": 66,
                "produto": "Máquininha Corta Barba",
                "data_compra": "2020-03-27T11:51:16"
            }
        }
    }


class LoteInput(BaseModel):
    """Lista de transações para avaliação em lote."""

    transacoes: list[TransacaoInput] = Field(
        ...,
        min_length=1,
        description="Lista de transações. Mínimo: 1."
    )


# ══════════════════════════════════════════════════════════════════
#  SAÍDA
# ══════════════════════════════════════════════════════════════════

class PredicaoOutput(BaseModel):
    """Resultado da predição para uma única transação."""

    fraude:       int   = Field(..., description="0 = legítima | 1 = fraude")
    probabilidade: float = Field(..., description="Probabilidade de fraude (0 a 1)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "fraude": 1,
                "probabilidade": 0.8731
            }
        }
    }


class ItemLoteOutput(BaseModel):
    """Resultado de um item dentro de uma resposta de lote."""

    indice:        int
    fraude:        int
    probabilidade: float


class LoteOutput(BaseModel):
    """Resultado da predição para um lote de transações."""

    total:             int
    total_fraudes:     int
    taxa_fraude:       float = Field(..., description="Percentual de fraudes detectadas")
    predicoes:         list[ItemLoteOutput]


class HealthOutput(BaseModel):
    """Resposta do endpoint de saúde."""

    status:      str
    modelo:      str
    threshold:   float
    versao_api:  str