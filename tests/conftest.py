"""
conftest.py
===========
Fixtures compartilhadas entre todos os testes.

O DataFrame sintético imita a estrutura exata do fraude.xlsx, mas usa
dados fictícios — assim os testes rodam em qualquer ambiente, inclusive
no CI/CD, sem depender do arquivo real.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def df_sintetico() -> pd.DataFrame:
    """DataFrame sintético com a estrutura exata do fraude.xlsx.

    - 200 linhas (190 legítimas + 10 fraudes) para cobrir os dois casos.
    - Colunas na mesma ordem que o pipeline espera.
    - Inclui propositalmente NaN em 'entrega_doc_2' e 'pais' para testar
      os imputers.
    - Inclui as colunas extras que o pipeline remove internamente
      (score_fraude_modelo, produto, data_compra).
    """
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame({
        # ── Ordem exata esperada pelo pipeline ────────────────────────
        'score_1':             rng.integers(1, 10, size=n).astype('int64'),
        'score_2':             rng.uniform(0.0, 1.0, size=n).astype('float64'),
        'score_3':             rng.uniform(100.0, 100_000.0, size=n).astype('float64'),
        'score_4':             rng.uniform(0.0, 50.0, size=n).astype('float64'),
        'score_5':             rng.uniform(0.0, 1.0, size=n).astype('float64'),
        'score_6':             rng.uniform(0.0, 100.0, size=n).astype('float64'),
        'pais':                _com_nulos(
                                   rng.choice(['BR', 'US', 'AR', 'MX'], size=n),
                                   rng, frac=0.05
                               ),
        'score_7':             rng.integers(0, 10, size=n).astype('int64'),
        'categoria_produto':   rng.choice([f'cat_{i}' for i in range(10)], size=n),
        'score_8':             rng.uniform(0.0, 1.0, size=n).astype('float64'),
        'score_9':             rng.uniform(0.0, 5000.0, size=n).astype('float64'),
        'score_10':            rng.uniform(0.0, 200.0, size=n).astype('float64'),
        'entrega_doc_1':       rng.integers(0, 2, size=n).astype('int64'),
        'entrega_doc_2':       _com_nulos(
                                   rng.choice(['Y', 'N'], size=n),
                                   rng, frac=0.10
                               ),
        'entrega_doc_3':       rng.choice(['Y', 'N'], size=n),
        'valor_compra':        rng.uniform(5.0, 5000.0, size=n).astype('float64'),

        # ── Colunas que o pipeline remove internamente ─────────────────
        'score_fraude_modelo': rng.integers(0, 100, size=n).astype('int64'),
        'produto':             [f'Produto_{i}' for i in range(n)],
        'data_compra':         pd.date_range(start='2020-01-01', periods=n, freq='D'),

        # ── Alvo ───────────────────────────────────────────────────────
        'fraude':              _gerar_alvo(n, n_fraudes=10),
    })

    return df


@pytest.fixture(scope="session")
def df_sem_alvo(df_sintetico) -> pd.DataFrame:
    """Mesmo DataFrame sintético, mas sem a coluna 'fraude'."""
    return df_sintetico.drop(columns=['fraude'])


# ── Helpers internos ───────────────────────────────────────────────────────────

def _com_nulos(arr: np.ndarray, rng: np.random.Generator, frac: float) -> np.ndarray:
    """Insere NaN aleatoriamente em uma fração dos elementos."""
    result = arr.astype(object)
    idx = rng.choice(len(result), size=int(len(result) * frac), replace=False)
    result[idx] = np.nan
    return result


def _gerar_alvo(n: int, n_fraudes: int) -> np.ndarray:
    """Gera vetor alvo com n_fraudes fraudes nas últimas posições."""
    alvo = np.zeros(n, dtype='int64')
    alvo[-n_fraudes:] = 1
    return alvo