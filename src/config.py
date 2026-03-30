"""
config.py
=========
Centraliza caminhos, constantes e configurações do projeto.
Todas as outras classes importam daqui — nunca usam strings hard-coded.
"""

from pathlib import Path

# ── Raiz do projeto ────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# ── Diretórios principais ──────────────────────────────────────────────────────
DATA_DIR: Path   = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"

# ── Artefatos ──────────────────────────────────────────────────────────────────
PIPELINE_PATH: Path = MODELS_DIR / "pipeline_fraude_completo.joblib"
DATA_PATH: Path     = DATA_DIR   / "fraude.xlsx"

# ── Configurações da simulação ─────────────────────────────────────────────────
SIMULATION_N_ROWS: int  = 1_000
SIMULATION_SEED: int    = 42

# ── Coluna alvo ────────────────────────────────────────────────────────────────
TARGET_COL: str = "fraude"

# ── Features contínuas (usadas no pipeline de transformação) ───────────────────
CONTINUOUS_FEATURES: list[str] = [
    "score_1", "score_2", "score_3", "score_4", "score_5",
    "score_6", "score_7", "score_8", "score_9", "score_10",
    "entrega_doc_1", "entrega_doc_2", "valor_compra",
    "doc_2_vazio", "entrega_doc_3",
]

# ── Features selecionadas pelo Boruta (entrada do modelo) ─────────────────────
SELECTED_FEATURES: list[str] = [
    "pais_kmeans", "score_9 entrega_doc_1", "entrega_doc_2 valor_compra",
    "valor_compra", "entrega_doc_1^2", "score_6", "score_9 entrega_doc_2",
    "score_10 entrega_doc_1", "entrega_doc_1 entrega_doc_2", "entrega_doc_1_kmeans",
    "entrega_doc_2 doc_2_vazio", "score_1 score_2", "score_10_kmeans",
    "score_1 valor_compra", "doc_2_vazio entrega_doc_3", "doc_2_vazio^2",
    "score_7 score_10", "score_7 entrega_doc_1", "score_9_kmeans", "pais",
    "categoria_produto_kmeans", "score_4 score_9", "doc_2_vazio",
    "entrega_doc_1 entrega_doc_3", "doc_2_vazio_kmeans", "score_1",
    "score_1 score_7", "score_1 entrega_doc_3", "score_9 entrega_doc_3",
    "score_9 score_10", "score_6 entrega_doc_2", "entrega_doc_2", "score_9",
    "score_10", "entrega_doc_1", "score_1_kmeans", "categoria_produto",
    "score_1 entrega_doc_2", "entrega_doc_3", "score_7 valor_compra",
    "score_6 entrega_doc_1", "score_2", "score_1 entrega_doc_1",
    "score_2_kmeans", "score_2^2", "score_6 score_10", "score_4",
]