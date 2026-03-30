"""
pipeline_fraude.py
==================
Transformadores customizados do pipeline sklearn de detecção de fraudes.

Todas as classes seguem a interface BaseEstimator + TransformerMixin do
scikit-learn, garantindo compatibilidade com Pipeline, GridSearchCV e joblib.

Classes
-------
DropColumnsTransformer        : Remove colunas desnecessárias e converte dtypes.
Doc2Transformer               : Engenharia de features sobre entrega_doc_2.
PaisImputerTransformer        : Imputa coluna 'pais' com a moda do treino.
CategoriaProdutoGrouper       : Agrupa categorias raras em 'Outros'.
KNNImputerNumerico            : KNN imputer para colunas numéricas.
CatBoostEncoderTransformer    : CatBoost Encoding para colunas categóricas.
KMeansDiscretizerTransformer  : Discretização KMeans com bins ótimos (silhouette).
StandardScalerTransformer     : StandardScaler preservando nomes de colunas.
PolynomialFeaturesTransformer : Features polinomiais grau 2 nas features contínuas.
FeatureSelectorTransformer    : Seleciona as features escolhidas pelo Boruta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import (
    KBinsDiscretizer,
    PolynomialFeatures,
    StandardScaler,
)
from category_encoders import CatBoostEncoder

from .config import CONTINUOUS_FEATURES, SELECTED_FEATURES

warnings.filterwarnings("ignore")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_optimal_bins(
    X_col: np.ndarray,
    min_bins: int = 2,
    max_bins: int = 5,
    random_state: int = 42,
    max_samples: int = 10_000,
) -> int:
    """Retorna o número ótimo de bins via silhouette score."""
    scores: dict[int, float] = {}

    for n in range(min_bins, max_bins + 1):
        disc = KBinsDiscretizer(
            n_bins=n,
            encode="ordinal",
            strategy="kmeans",
            random_state=random_state,
        )
        labels = disc.fit_transform(X_col).ravel()
        n_unique = len(np.unique(labels))

        if 2 <= n_unique < X_col.shape[0]:
            try:
                if X_col.shape[0] > max_samples:
                    rng = np.random.RandomState(random_state)
                    idx = rng.choice(X_col.shape[0], size=max_samples, replace=False)
                    score = silhouette_score(X_col[idx], labels[idx])
                else:
                    score = silhouette_score(X_col, labels)
                scores[n] = score
            except Exception:
                scores[n] = -1.0
        else:
            scores[n] = -1.0

    return max(scores, key=scores.get)


# ── Transformadores ────────────────────────────────────────────────────────────

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Remove colunas desnecessárias e converte float64 → float32.

    Parameters
    ----------
    cols_to_drop : list[str], optional
        Colunas a remover. Padrão: ['score_fraude_modelo', 'produto', 'data_compra'].
    """

    def __init__(
        self,
        cols_to_drop: list[str] | None = None,
    ) -> None:
        self.cols_to_drop = cols_to_drop or [
            "score_fraude_modelo",
            "produto",
            "data_compra",
        ]

    def fit(self, X: pd.DataFrame, y=None) -> "DropColumnsTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols_existentes = [c for c in self.cols_to_drop if c in X.columns]
        X = X.drop(columns=cols_existentes)
        float64_cols = X.select_dtypes(include=["float64"]).columns
        X[float64_cols] = X[float64_cols].astype("float32")
        return X


class Doc2Transformer(BaseEstimator, TransformerMixin):
    """Cria 'doc_2_vazio' e converte 'entrega_doc_2' de Y/N → 0/1."""

    def fit(self, X: pd.DataFrame, y=None) -> "Doc2Transformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["doc_2_vazio"] = np.where(X["entrega_doc_2"].isna(), 1, 0)
        X["entrega_doc_2"] = (
            X["entrega_doc_2"]
            .map({"Y": 1, "N": 0})
            .fillna(0)
            .astype(int)
        )
        return X


class PaisImputerTransformer(BaseEstimator, TransformerMixin):
    """Imputa coluna 'pais' com a moda calculada no treino."""

    def __init__(self) -> None:
        self.imputer_: SimpleImputer | None = None

    def fit(self, X: pd.DataFrame, y=None) -> "PaisImputerTransformer":
        self.imputer_ = SimpleImputer(strategy="most_frequent")
        self.imputer_.fit(X[["pais"]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["pais"] = self.imputer_.transform(X[["pais"]]).ravel()
        return X


class CategoriaProdutoGrouper(BaseEstimator, TransformerMixin):
    """Agrupa categorias raras de 'categoria_produto' em 'Outros'.

    Mantém apenas as categorias que acumulam 80% das fraudes do treino.
    """

    def __init__(self) -> None:
        self.categorias_principais_: list[str] = []

    def fit(self, X: pd.DataFrame, y) -> "CategoriaProdutoGrouper":
        df_train = X.copy()
        df_train["fraude"] = y.values if hasattr(y, "values") else y

        fraudes = (
            df_train[df_train["fraude"] == 1]["categoria_produto"]
            .value_counts()
        )
        pct_acum = (fraudes.cumsum() / fraudes.sum()) * 100
        cats = pct_acum[pct_acum <= 80].index.tolist()

        if len(cats) < len(pct_acum):
            cats.append(pct_acum.index[len(cats)])

        self.categorias_principais_ = cats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["categoria_produto"] = X["categoria_produto"].apply(
            lambda v: v if v in self.categorias_principais_ else "Outros"
        )
        return X


class KNNImputerNumerico(BaseEstimator, TransformerMixin):
    """KNN imputer aplicado apenas às colunas numéricas.

    Parameters
    ----------
    n_neighbors : int
        Número de vizinhos. Padrão: 5.
    weights : str
        Estratégia de pesos ('uniform' ou 'distance'). Padrão: 'distance'.
    chunk_size : int
        Tamanho do chunk para imputação em lotes. Padrão: 100.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        chunk_size: int = 100,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.chunk_size = chunk_size

    def fit(self, X: pd.DataFrame, y=None) -> "KNNImputerNumerico":
        self.num_cols_: list[str] = X.select_dtypes(
            include=["float32", "float64", "int64", "int32"]
        ).columns.tolist()

        self.imputer_ = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric="nan_euclidean",
        )
        self.imputer_.fit(X[self.num_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        chunk_df_list = []
        cs = self.chunk_size
        X_num = X[self.num_cols_]

        for i in range(0, len(X_num), cs):
            chunk = X_num.iloc[i : i + cs]
            transformed = self.imputer_.transform(chunk)
            chunk_df_list.append(
                pd.DataFrame(transformed, columns=self.num_cols_, index=chunk.index)
            )

        X[self.num_cols_] = pd.concat(chunk_df_list)
        return X


class CatBoostEncoderTransformer(BaseEstimator, TransformerMixin):
    """CatBoost Encoding para colunas categóricas (object).

    Parameters
    ----------
    a : float
        Fator de suavização. Padrão: 1000.
    """

    def __init__(self, a: float = 1000) -> None:
        self.a = a

    def fit(self, X: pd.DataFrame, y) -> "CatBoostEncoderTransformer":
        self.cat_cols_: list[str] = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        X_cat = X[self.cat_cols_].astype(str)
        self.encoder_ = CatBoostEncoder(cols=self.cat_cols_, a=self.a)
        self.encoder_.fit(X_cat, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X_cat = X[self.cat_cols_].astype(str)
        X_encoded = self.encoder_.transform(X_cat)
        X[self.cat_cols_] = X_encoded[self.cat_cols_].values
        return X


class KMeansDiscretizerTransformer(BaseEstimator, TransformerMixin):
    """Discretização KMeans com número ótimo de bins por coluna (silhouette).

    Parameters
    ----------
    min_bins : int
        Mínimo de bins. Padrão: 2.
    max_bins : int
        Máximo de bins. Padrão: 5.
    random_state : int
        Semente aleatória. Padrão: 42.
    max_samples : int
        Limite de amostras para cálculo do silhouette. Padrão: 10_000.
    """

    def __init__(
        self,
        min_bins: int = 2,
        max_bins: int = 5,
        random_state: int = 42,
        max_samples: int = 10_000,
    ) -> None:
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.random_state = random_state
        self.max_samples = max_samples

    def fit(self, X: pd.DataFrame, y=None) -> "KMeansDiscretizerTransformer":
        self.cols_: list[str] = X.columns.tolist()
        X_arr = X.values
        self.discretizers_: list[KBinsDiscretizer] = []

        print("  [KMeansDiscretizer] Buscando bins ótimos por coluna...")
        for i, col in enumerate(tqdm(self.cols_, desc="  Bins ótimos")):
            col_arr = X_arr[:, i].reshape(-1, 1)
            n_opt = _find_optimal_bins(
                col_arr,
                self.min_bins,
                self.max_bins,
                self.random_state,
                self.max_samples,
            )
            disc = KBinsDiscretizer(
                n_bins=n_opt,
                encode="ordinal",
                strategy="kmeans",
                random_state=self.random_state,
            )
            disc.fit(col_arr)
            self.discretizers_.append(disc)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X_arr = X[self.cols_].values

        for i, col in enumerate(self.cols_):
            col_arr = X_arr[:, i].reshape(-1, 1)
            X[f"{col}_kmeans"] = (
                self.discretizers_[i].transform(col_arr).ravel()
            )
        return X


class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    """StandardScaler que preserva o DataFrame com nomes de colunas."""

    def __init__(self) -> None:
        self.scaler_: StandardScaler | None = None
        self.cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "StandardScalerTransformer":
        self.cols_ = X.columns.tolist()
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.scaler_.transform(X)
        return pd.DataFrame(arr, columns=self.cols_, index=X.index)


class PolynomialFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Features polinomiais grau 2 nas CONTINUOUS_FEATURES, sem duplicar originais.

    Parameters
    ----------
    degree : int
        Grau polinomial. Padrão: 2.
    continuous_features : list[str], optional
        Features contínuas base. Padrão: CONTINUOUS_FEATURES do config.
    """

    def __init__(
        self,
        degree: int = 2,
        continuous_features: list[str] | None = None,
    ) -> None:
        self.degree = degree
        self.continuous_features = continuous_features or CONTINUOUS_FEATURES

    def fit(self, X: pd.DataFrame, y=None) -> "PolynomialFeaturesTransformer":
        self.present_cont_: list[str] = [
            c for c in self.continuous_features if c in X.columns
        ]
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=False,
            include_bias=False,
        )
        self.poly_.fit(X[self.present_cont_])

        all_names = self.poly_.get_feature_names_out(self.present_cont_)
        self.new_feature_names_: list[str] = [
            n for n in all_names if n not in self.present_cont_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        poly_arr = self.poly_.transform(X[self.present_cont_])
        all_names = self.poly_.get_feature_names_out(self.present_cont_)
        poly_df = pd.DataFrame(poly_arr, columns=all_names, index=X.index)
        return pd.concat([X, poly_df[self.new_feature_names_]], axis=1)


class FeatureSelectorTransformer(BaseEstimator, TransformerMixin):
    """Seleciona apenas as features escolhidas pelo Boruta.

    Parameters
    ----------
    selected_features : list[str], optional
        Lista de features. Padrão: SELECTED_FEATURES do config.
    """

    def __init__(
        self,
        selected_features: list[str] | None = None,
    ) -> None:
        self.selected_features = selected_features or SELECTED_FEATURES

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelectorTransformer":
        self.features_present_: list[str] = [
            f for f in self.selected_features if f in X.columns
        ]
        missing = set(self.selected_features) - set(self.features_present_)
        if missing:
            print(f"  [FeatureSelector] ATENÇÃO — features ausentes: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features_present_]