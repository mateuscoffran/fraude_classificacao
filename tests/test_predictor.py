"""
test_predictor.py
=================
Testes automatizados para a classe FraudPredictor.

Categorias
----------
- Carregamento     : o pipeline é carregado corretamente do disco?
- Contrato de saída: predict e predict_proba retornam o formato esperado?
- Negócio          : as predições fazem sentido para o problema?
- Robustez         : erros de input são tratados corretamente?
"""

import numpy as np
import pytest

from src.predictor import FraudPredictor


# ── Fixture: uma única instância do predictor para toda a sessão de testes ────
@pytest.fixture(scope="session")
def predictor() -> FraudPredictor:
    return FraudPredictor()


# ══════════════════════════════════════════════════════════════════
#  1. CARREGAMENTO
# ══════════════════════════════════════════════════════════════════

class TestCarregamento:

    def test_pipeline_carregado(self, predictor):
        """O pipeline não deve ser None após instanciar o FraudPredictor."""
        assert predictor.pipeline_ is not None

    def test_model_type_preenchido(self, predictor):
        """O tipo do modelo deve ser uma string não vazia."""
        assert isinstance(predictor.model_type_, str)
        assert len(predictor.model_type_) > 0

    def test_threshold_entre_zero_e_um(self, predictor):
        """O threshold otimizado deve estar entre 0 e 1."""
        assert 0.0 < predictor.threshold_ < 1.0

    def test_pipeline_nao_encontrado_lanca_erro(self):
        """Deve lançar FileNotFoundError para caminho inexistente."""
        with pytest.raises(FileNotFoundError):
            FraudPredictor(pipeline_path="caminho/inexistente.joblib")


# ══════════════════════════════════════════════════════════════════
#  2. CONTRATO DE SAÍDA — predict
# ══════════════════════════════════════════════════════════════════

class TestPredict:

    def test_retorna_numpy_array(self, predictor, df_sem_alvo):
        """predict deve retornar um numpy array."""
        y_pred = predictor.predict(df_sem_alvo)
        assert isinstance(y_pred, np.ndarray)

    def test_tamanho_igual_ao_input(self, predictor, df_sem_alvo):
        """O array de predições deve ter o mesmo número de linhas do input."""
        y_pred = predictor.predict(df_sem_alvo)
        assert len(y_pred) == len(df_sem_alvo)

    def test_valores_apenas_zero_ou_um(self, predictor, df_sem_alvo):
        """predict deve retornar apenas 0s e 1s."""
        y_pred = predictor.predict(df_sem_alvo)
        assert set(y_pred).issubset({0, 1})

    def test_funciona_com_uma_linha(self, predictor, df_sem_alvo):
        """O pipeline deve funcionar com apenas 1 linha de input."""
        y_pred = predictor.predict(df_sem_alvo.iloc[[0]])
        assert len(y_pred) == 1
        assert y_pred[0] in {0, 1}


# ══════════════════════════════════════════════════════════════════
#  3. CONTRATO DE SAÍDA — predict_proba
# ══════════════════════════════════════════════════════════════════

class TestPredictProba:

    def test_retorna_numpy_array(self, predictor, df_sem_alvo):
        """predict_proba deve retornar um numpy array."""
        y_proba = predictor.predict_proba(df_sem_alvo)
        assert isinstance(y_proba, np.ndarray)

    def test_tamanho_igual_ao_input(self, predictor, df_sem_alvo):
        """O array de probabilidades deve ter o mesmo número de linhas do input."""
        y_proba = predictor.predict_proba(df_sem_alvo)
        assert len(y_proba) == len(df_sem_alvo)

    def test_valores_entre_zero_e_um(self, predictor, df_sem_alvo):
        """Todas as probabilidades devem estar no intervalo [0, 1]."""
        y_proba = predictor.predict_proba(df_sem_alvo)
        assert np.all(y_proba >= 0.0)
        assert np.all(y_proba <= 1.0)

    def test_funciona_com_uma_linha(self, predictor, df_sem_alvo):
        """predict_proba deve funcionar com apenas 1 linha."""
        y_proba = predictor.predict_proba(df_sem_alvo.iloc[[0]])
        assert len(y_proba) == 1
        assert 0.0 <= y_proba[0] <= 1.0


# ══════════════════════════════════════════════════════════════════
#  4. CONTRATO DE SAÍDA — predict_with_proba
# ══════════════════════════════════════════════════════════════════

class TestPredictWithProba:

    def test_retorna_dataframe(self, predictor, df_sem_alvo):
        """predict_with_proba deve retornar um DataFrame."""
        import pandas as pd
        resultado = predictor.predict_with_proba(df_sem_alvo)
        assert isinstance(resultado, pd.DataFrame)

    def test_colunas_adicionadas(self, predictor, df_sem_alvo):
        """As colunas 'fraude_pred' e 'fraude_proba' devem ser adicionadas."""
        resultado = predictor.predict_with_proba(df_sem_alvo)
        assert "fraude_pred" in resultado.columns
        assert "fraude_proba" in resultado.columns

    def test_numero_de_linhas_preservado(self, predictor, df_sem_alvo):
        """O número de linhas do input deve ser preservado."""
        resultado = predictor.predict_with_proba(df_sem_alvo)
        assert len(resultado) == len(df_sem_alvo)

    def test_colunas_originais_preservadas(self, predictor, df_sem_alvo):
        """Todas as colunas originais do input devem estar no resultado."""
        resultado = predictor.predict_with_proba(df_sem_alvo)
        for col in df_sem_alvo.columns:
            assert col in resultado.columns


# ══════════════════════════════════════════════════════════════════
#  5. NEGÓCIO
# ══════════════════════════════════════════════════════════════════

class TestNegocio:

    def test_taxa_de_fraude_dentro_do_esperado(self, predictor, df_sem_alvo):
        """A taxa de fraude predita deve estar entre 1% e 50%.

        Um modelo que classifica tudo como fraude (100%) ou nada (0%)
        claramente não está funcionando corretamente.
        """
        y_pred = predictor.predict(df_sem_alvo)
        taxa = y_pred.mean()
        assert 0.01 <= taxa <= 0.50, (
            f"Taxa de fraude fora do esperado: {taxa:.2%}. "
            "Verifique se o pipeline está correto."
        )

    def test_modelo_detecta_alguma_fraude(self, predictor, df_sem_alvo):
        """O modelo deve detectar pelo menos uma fraude nas 200 linhas."""
        y_pred = predictor.predict(df_sem_alvo)
        assert y_pred.sum() >= 1, "O modelo não detectou nenhuma fraude."

    def test_probabilidade_media_razoavel(self, predictor, df_sem_alvo):
        """A probabilidade média de fraude deve estar entre 1% e 50%."""
        y_proba = predictor.predict_proba(df_sem_alvo)
        media = y_proba.mean()
        assert 0.01 <= media <= 0.50, (
            f"Probabilidade média fora do esperado: {media:.2%}."
        )


# ══════════════════════════════════════════════════════════════════
#  6. ROBUSTEZ — tratamento de inputs inválidos
# ══════════════════════════════════════════════════════════════════

class TestRobustez:

    def test_dataframe_vazio_lanca_erro(self, predictor, df_sem_alvo):
        """Deve lançar ValueError para DataFrame vazio."""
        import pandas as pd
        with pytest.raises(ValueError):
            predictor.predict(pd.DataFrame())

    def test_input_nao_dataframe_lanca_erro(self, predictor):
        """Deve lançar TypeError para input que não seja DataFrame."""
        with pytest.raises(TypeError):
            predictor.predict([[1, 2, 3]])