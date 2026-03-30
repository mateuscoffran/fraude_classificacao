"""
predictor.py
============
Responsável pelo carregamento do pipeline treinado e pela inferência.

Classe principal
----------------
FraudPredictor
    Carrega o pipeline_fraude_completo.joblib e expõe métodos de predição
    prontos para uso em produção.

Uso rápido
----------
    from src.predictor import FraudPredictor

    predictor = FraudPredictor()
    y_pred    = predictor.predict(df_raw)
    y_proba   = predictor.predict_proba(df_raw)
    resultado = predictor.predict_with_proba(df_raw)
"""

from __future__ import annotations

import logging
from pathlib import Path

import sys

import joblib
import pandas as pd
import numpy as np

from .config import PIPELINE_PATH
from . import pipeline_fraude
from .pipeline_fraude import (  # noqa: F401
    DropColumnsTransformer,
    Doc2Transformer,
    PaisImputerTransformer,
    CategoriaProdutoGrouper,
    KNNImputerNumerico,
    CatBoostEncoderTransformer,
    KMeansDiscretizerTransformer,
    StandardScalerTransformer,
    PolynomialFeaturesTransformer,
    FeatureSelectorTransformer,
)


# Mapa de todas as classes customizadas do pipeline
_CUSTOM_CLASSES: dict[str, type] = {
    "DropColumnsTransformer":        DropColumnsTransformer,
    "Doc2Transformer":               Doc2Transformer,
    "PaisImputerTransformer":        PaisImputerTransformer,
    "CategoriaProdutoGrouper":       CategoriaProdutoGrouper,
    "KNNImputerNumerico":            KNNImputerNumerico,
    "CatBoostEncoderTransformer":    CatBoostEncoderTransformer,
    "KMeansDiscretizerTransformer":  KMeansDiscretizerTransformer,
    "StandardScalerTransformer":     StandardScalerTransformer,
    "PolynomialFeaturesTransformer": PolynomialFeaturesTransformer,
    "FeatureSelectorTransformer":    FeatureSelectorTransformer,
}


def _patch_main_classes() -> None:
    """Injeta as classes customizadas em sys.modules['__main__'].

    Quando o pipeline é salvo num notebook, o pickle registra as classes
    com módulo '__main__'. Ao carregar em outro contexto, o pickle procura
    '__main__.NomeDaClasse' — que não existe. Injetamos as classes
    diretamente em __main__ antes do joblib.load para que o pickle as
    encontre normalmente, sem precisar interceptar o stream comprimido.
    """
    main_module = sys.modules.setdefault("__main__", sys.modules[__name__])
    for name, cls in _CUSTOM_CLASSES.items():
        if not hasattr(main_module, name):
            setattr(main_module, name, cls)


def _joblib_load_remapped(path):
    """Carrega o pipeline garantindo que as classes customizadas estejam
    disponíveis em __main__ antes da desserialização.
    """
    _patch_main_classes()
    return joblib.load(path)

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Carrega o pipeline completo e realiza inferências sobre dados brutos.

    O pipeline internamente executa todas as etapas de pré-processamento
    antes de chamar o classificador, portanto o DataFrame de entrada deve
    estar no formato original (igual ao fraude.xlsx), sem nenhuma
    transformação prévia.

    Parameters
    ----------
    pipeline_path : Path | str, optional
        Caminho para o arquivo .joblib. Padrão: PIPELINE_PATH do config.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        Pipeline completo carregado.
    threshold_ : float
        Threshold otimizado pelo TunedThresholdClassifierCV.
    model_type_ : str
        Nome da classe do estimador interno.

    Examples
    --------
    >>> predictor = FraudPredictor()
    >>> y_pred = predictor.predict(df_raw)
    >>> resultado = predictor.predict_with_proba(df_raw)
    """

    def __init__(
        self,
        pipeline_path: Path | str | None = None,
    ) -> None:
        self._pipeline_path = Path(pipeline_path or PIPELINE_PATH)
        self.pipeline_       = None
        self.threshold_      = None
        self.model_type_     = None
        self._load()

    # ── Carregamento ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Carrega o pipeline do disco e extrai metadados do modelo."""
        if not self._pipeline_path.exists():
            raise FileNotFoundError(
                f"Pipeline não encontrado em: {self._pipeline_path}\n"
                "Verifique se o arquivo .joblib está na pasta 'models/'."
            )

        logger.info("Carregando pipeline de '%s'...", self._pipeline_path)
        self.pipeline_ = _joblib_load_remapped(self._pipeline_path)

        # Extrai threshold e tipo do modelo interno
        model_step = self.pipeline_.named_steps.get("model")
        if model_step is None:
            raise ValueError(
                "O pipeline carregado não possui o step 'model'. "
                "Verifique a estrutura do .joblib."
            )

        # TunedThresholdClassifierCV expõe best_threshold_
        if hasattr(model_step, "best_threshold_"):
            self.threshold_ = model_step.best_threshold_
            inner = getattr(model_step, "estimator", model_step)
            self.model_type_ = type(inner).__name__
        else:
            self.threshold_ = 0.5
            self.model_type_ = type(model_step).__name__

        logger.info(
            "Pipeline carregado — modelo: %s | threshold: %.4f",
            self.model_type_,
            self.threshold_,
        )

    # ── Interface pública ──────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna as classes preditas (0 = legítimo, 1 = fraude).

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame bruto no formato original da base fraude.xlsx.

        Returns
        -------
        np.ndarray
            Array de inteiros com as predições.
        """
        self._validate_input(X)
        return self.pipeline_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna as probabilidades de fraude (classe 1).

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame bruto no formato original.

        Returns
        -------
        np.ndarray
            Array 1-D com a probabilidade de fraude para cada linha.
        """
        self._validate_input(X)
        return self.pipeline_.predict_proba(X)[:, 1]

    def predict_with_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retorna o DataFrame original acrescido de colunas de resultado.

        Colunas adicionadas
        -------------------
        fraude_pred  : int   — classe predita (0 ou 1)
        fraude_proba : float — probabilidade de fraude

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame bruto no formato original.

        Returns
        -------
        pd.DataFrame
            Cópia do DataFrame de entrada com as colunas de resultado.
        """
        self._validate_input(X)
        result = X.copy()
        result["fraude_pred"]  = self.predict(X)
        result["fraude_proba"] = self.predict_proba(X)
        return result

    # ── Validação ──────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        """Valida o tipo e o tamanho mínimo do input."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"O input deve ser um pd.DataFrame, recebido: {type(X).__name__}"
            )
        if X.empty:
            raise ValueError("O DataFrame de entrada está vazio.")

    # ── Repr ───────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "carregado" if self.pipeline_ is not None else "não carregado"
        return (
            f"FraudPredictor("
            f"model={self.model_type_}, "
            f"threshold={self.threshold_:.4f}, "
            f"status={status})"
        )