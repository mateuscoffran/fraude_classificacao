"""
test_simulator.py
=================
Testes automatizados para a classe FraudSimulator.

Como o FraudSimulator carrega o fraude.xlsx internamente, usamos
monkeypatch para substituir o método _load_sample pelo nosso DataFrame
sintético — sem depender do arquivo real em nenhum momento.

Categorias
----------
- Execução   : o run() completa sem erros?
- Saída      : o resultado tem a estrutura esperada?
- Filtragem  : fraude_detected contém apenas fraudes preditas?
- Relatório  : report() executa sem erros?
"""

import pandas as pd
import pytest

from src.simulator import FraudSimulator


# ── Fixture: simulador com _load_sample substituído pelo df sintético ─────────

@pytest.fixture(scope="session")
def simulator_com_dados_sinteticos(df_sintetico) -> FraudSimulator:
    """Instancia o FraudSimulator e injeta o DataFrame sintético,
    evitando qualquer leitura do fraude.xlsx real.
    """
    sim = FraudSimulator()
    # Substitui a amostra diretamente — pula o _load_sample
    sim.sample_ = df_sintetico.copy()
    sim._predict()
    sim._filter_fraud()
    return sim


# ══════════════════════════════════════════════════════════════════
#  1. EXECUÇÃO
# ══════════════════════════════════════════════════════════════════

class TestExecucao:

    def test_run_completa_sem_erros(self, tmp_path, df_sintetico, monkeypatch):
        """run() deve completar sem lançar exceções com dados sintéticos."""
        # Salva o df sintético como xlsx temporário
        data_file = tmp_path / "fraude.xlsx"
        df_sintetico.to_excel(data_file, index=False)

        sim = FraudSimulator(data_path=data_file)
        resultado = sim.run()
        assert resultado is not None

    def test_predict_preenche_results(self, simulator_com_dados_sinteticos):
        """Após _predict(), results_ não deve ser None."""
        assert simulator_com_dados_sinteticos.results_ is not None

    def test_filter_preenche_fraude_detected(self, simulator_com_dados_sinteticos):
        """Após _filter_fraud(), fraude_detected não deve ser None."""
        assert simulator_com_dados_sinteticos.fraude_detected is not None


# ══════════════════════════════════════════════════════════════════
#  2. ESTRUTURA DA SAÍDA
# ══════════════════════════════════════════════════════════════════

class TestEstruturaSaida:

    def test_results_e_dataframe(self, simulator_com_dados_sinteticos):
        """results_ deve ser um DataFrame."""
        assert isinstance(simulator_com_dados_sinteticos.results_, pd.DataFrame)

    def test_fraude_detected_e_dataframe(self, simulator_com_dados_sinteticos):
        """fraude_detected deve ser um DataFrame."""
        assert isinstance(simulator_com_dados_sinteticos.fraude_detected, pd.DataFrame)

    def test_coluna_fraude_pred_existe(self, simulator_com_dados_sinteticos):
        """results_ deve conter a coluna 'fraude_pred'."""
        assert "fraude_pred" in simulator_com_dados_sinteticos.results_.columns

    def test_coluna_fraude_proba_existe(self, simulator_com_dados_sinteticos):
        """results_ deve conter a coluna 'fraude_proba'."""
        assert "fraude_proba" in simulator_com_dados_sinteticos.results_.columns

    def test_numero_de_linhas_em_results(self, simulator_com_dados_sinteticos, df_sintetico):
        """results_ deve ter o mesmo número de linhas do df sintético."""
        assert len(simulator_com_dados_sinteticos.results_) == len(df_sintetico)

    def test_fraude_proba_entre_zero_e_um(self, simulator_com_dados_sinteticos):
        """Todas as probabilidades em results_ devem estar no intervalo [0, 1]."""
        proba = simulator_com_dados_sinteticos.results_["fraude_proba"]
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()


# ══════════════════════════════════════════════════════════════════
#  3. FILTRAGEM
# ══════════════════════════════════════════════════════════════════

class TestFiltragem:

    def test_fraude_detected_contem_apenas_fraudes_preditas(
        self, simulator_com_dados_sinteticos
    ):
        """Todas as linhas em fraude_detected devem ter fraude_pred == 1."""
        fd = simulator_com_dados_sinteticos.fraude_detected
        if not fd.empty:
            assert (fd["fraude_pred"] == 1).all()

    def test_fraude_detected_ordenado_por_proba_decrescente(
        self, simulator_com_dados_sinteticos
    ):
        """fraude_detected deve estar ordenado por fraude_proba decrescente."""
        fd = simulator_com_dados_sinteticos.fraude_detected
        if len(fd) > 1:
            probas = fd["fraude_proba"].values
            assert all(probas[i] >= probas[i + 1] for i in range(len(probas) - 1))

    def test_fraude_detected_subconjunto_de_results(
        self, simulator_com_dados_sinteticos
    ):
        """fraude_detected não pode ter mais linhas do que results_."""
        n_results = len(simulator_com_dados_sinteticos.results_)
        n_detected = len(simulator_com_dados_sinteticos.fraude_detected)
        assert n_detected <= n_results


# ══════════════════════════════════════════════════════════════════
#  4. RELATÓRIO
# ══════════════════════════════════════════════════════════════════

class TestRelatorio:

    def test_report_executa_sem_erros(
        self, simulator_com_dados_sinteticos, capsys
    ):
        """report() deve executar sem lançar exceções."""
        simulator_com_dados_sinteticos.report()
        saida = capsys.readouterr().out
        assert len(saida) > 0

    def test_report_sem_run_imprime_aviso(self, capsys):
        """report() antes de run() deve imprimir mensagem de aviso."""
        sim = FraudSimulator()
        sim.report()
        saida = capsys.readouterr().out
        assert "run()" in saida