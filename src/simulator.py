"""
simulator.py
============
Simulação de inferência com dados reais da base de fraudes.

Classe principal
----------------
FraudSimulator
    Amostra N linhas da base original (sem injeção de anomalias),
    executa o pipeline e retorna o DataFrame 'fraude_detected' com
    apenas as linhas identificadas como fraude.

Uso rápido
----------
    from src.simulator import FraudSimulator

    sim = FraudSimulator()
    fraude_detected = sim.run()
    sim.report()
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import (
    DATA_PATH,
    PIPELINE_PATH,
    SIMULATION_N_ROWS,
    SIMULATION_SEED,
    TARGET_COL,
)
from .predictor import FraudPredictor

logger = logging.getLogger(__name__)


class FraudSimulator:
    """Simula inferência em produção usando dados reais da base original.

    Fluxo
    -----
    1. Carrega fraude.xlsx
    2. Amostra N linhas aleatórias (sem reposição, sem anomalias artificiais)
    3. Remove a coluna alvo antes de passar ao modelo
    4. Executa predict + predict_proba via FraudPredictor
    5. Retorna 'fraude_detected': apenas as linhas classificadas como fraude

    Parameters
    ----------
    data_path : Path | str, optional
        Caminho para fraude.xlsx. Padrão: DATA_PATH do config.
    pipeline_path : Path | str, optional
        Caminho para o .joblib. Padrão: PIPELINE_PATH do config.
    n_rows : int, optional
        Número de linhas a amostrar. Padrão: 1_000.
    seed : int, optional
        Semente aleatória para reprodutibilidade. Padrão: 42.

    Attributes
    ----------
    sample_          : pd.DataFrame  — amostra bruta (com coluna alvo, se existir)
    results_         : pd.DataFrame  — amostra com colunas fraude_pred e fraude_proba
    fraude_detected  : pd.DataFrame  — linhas classificadas como fraude

    Examples
    --------
    >>> sim = FraudSimulator()
    >>> fraude_detected = sim.run()
    >>> sim.report()
    """

    def __init__(
        self,
        data_path: Path | str | None = None,
        pipeline_path: Path | str | None = None,
        n_rows: int = SIMULATION_N_ROWS,
        seed: int = SIMULATION_SEED,
    ) -> None:
        self._data_path     = Path(data_path or DATA_PATH)
        self._pipeline_path = Path(pipeline_path or PIPELINE_PATH)
        self.n_rows         = n_rows
        self.seed           = seed

        self.predictor_     = FraudPredictor(pipeline_path=self._pipeline_path)
        self.sample_        = None
        self.results_       = None
        self.fraude_detected = None

    # ── Execução principal ─────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Executa a simulação completa e retorna 'fraude_detected'.

        Returns
        -------
        pd.DataFrame
            Linhas da amostra classificadas como fraude, com colunas
            'fraude_pred' e 'fraude_proba' adicionadas.
        """
        self._load_sample()
        self._predict()
        self._filter_fraud()
        self._log_summary()
        return self.fraude_detected

    # ── Etapas internas ────────────────────────────────────────────────────────

    def _load_sample(self) -> None:
        """Carrega e amostra N linhas da base original."""
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"Base de dados não encontrada em: {self._data_path}\n"
                "Verifique se fraude.xlsx está na pasta 'data/'."
            )

        logger.info("Carregando base de dados de '%s'...", self._data_path)
        df = pd.read_excel(self._data_path)

        n = min(self.n_rows, len(df))
        if n < self.n_rows:
            logger.warning(
                "Base possui apenas %d linhas; amostrando %d.", len(df), n
            )

        self.sample_ = df.sample(n=n, random_state=self.seed).reset_index(drop=True)
        logger.info("Amostra gerada: %d linhas × %d colunas.", *self.sample_.shape)

    def _predict(self) -> None:
        """Executa o pipeline sobre a amostra (sem a coluna alvo)."""
        # Remove a coluna alvo se presente — o modelo não a recebe
        X = self.sample_.drop(columns=[TARGET_COL], errors="ignore")
        self.results_ = self.predictor_.predict_with_proba(X)

        # Reinsere a coluna alvo original para permitir comparação posterior
        if TARGET_COL in self.sample_.columns:
            self.results_.insert(0, TARGET_COL, self.sample_[TARGET_COL].values)

    def _filter_fraud(self) -> None:
        """Filtra apenas as linhas preditas como fraude."""
        self.fraude_detected = (
            self.results_[self.results_["fraude_pred"] == 1]
            .sort_values("fraude_proba", ascending=False)
            .reset_index(drop=True)
        )

    def _log_summary(self) -> None:
        total     = len(self.results_)
        detectado = len(self.fraude_detected)
        taxa      = detectado / total * 100 if total else 0.0
        logger.info(
            "Simulação concluída — %d/%d fraudes detectadas (%.1f%%).",
            detectado, total, taxa,
        )

    # ── Relatório ──────────────────────────────────────────────────────────────

    def report(self) -> None:
        """Imprime um resumo da simulação no stdout."""
        if self.results_ is None:
            print("Execute .run() antes de chamar .report().")
            return

        total     = len(self.results_)
        detectado = len(self.fraude_detected)
        taxa      = detectado / total * 100 if total else 0.0

        sep = "=" * 55

        print(sep)
        print("  RELATÓRIO DE SIMULAÇÃO — DETECTOR DE FRAUDES")
        print(sep)
        print(f"  Modelo          : {self.predictor_.model_type_}")
        print(f"  Threshold       : {self.predictor_.threshold_:.4f}")
        print(f"  Linhas amostradas: {total:,}")
        print(f"  Fraudes detectadas: {detectado:,} ({taxa:.1f}%)")

        if TARGET_COL in self.results_.columns:
            real_fraudes = self.results_[TARGET_COL].sum()
            print(f"  Fraudes reais na amostra: {int(real_fraudes):,}")

            verdadeiros_positivos = (
                (self.fraude_detected[TARGET_COL] == 1).sum()
                if TARGET_COL in self.fraude_detected.columns
                else "N/A"
            )
            print(f"  Verdadeiros positivos  : {verdadeiros_positivos}")

        print(sep)

        if not self.fraude_detected.empty:
            cols_show = [
                c for c in [
                    TARGET_COL, "fraude_pred", "fraude_proba",
                    "valor_compra", "categoria_produto", "pais",
                ]
                if c in self.fraude_detected.columns
            ]
            print("\n  Top 10 maiores probabilidades de fraude:")
            print(
                self.fraude_detected[cols_show]
                .head(10)
                .to_string(index=True)
            )
        print(sep)

    # ── Repr ───────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "executada" if self.results_ is not None else "não executada"
        return (
            f"FraudSimulator("
            f"n_rows={self.n_rows}, "
            f"seed={self.seed}, "
            f"status={status})"
        )