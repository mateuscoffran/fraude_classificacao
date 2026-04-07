"""
locustfile.py
=============
Teste de carga da API de detecção de fraudes usando Locust.

Como rodar:
    # Sem Kubernetes (API rodando via uvicorn ou docker compose):
    locust -f locustfile.py --host=http://localhost:8000

    # Com Kubernetes (API rodando via Kind):
    locust -f locustfile.py --host=http://localhost:8080

Acesse http://localhost:8089 para a interface web do Locust.
"""

from locust import HttpUser, task, between


# Payload baseado exatamente no schema da API (TransacaoInput)
TRANSACAO_EXEMPLO = {
    "score_1":             4,
    "score_2":             0.7685,
    "score_3":             94436.24,
    "score_4":             20.0,
    "score_5":             0.44,
    "score_6":             1.0,
    "pais":                "BR",
    "score_7":             5,
    "categoria_produto":   "cat_8d714cd",
    "score_8":             0.88,
    "score_9":             240.0,
    "score_10":            102.0,
    "entrega_doc_1":       1,
    "entrega_doc_2":       None,
    "entrega_doc_3":       "N",
    "valor_compra":        5.64,
    "score_fraude_modelo": 66,
    "produto":             "Máquininha Corta Barba",
    "data_compra":         "2020-03-27T11:51:16"
}

LOTE_EXEMPLO = {
    "transacoes": [TRANSACAO_EXEMPLO] * 10
}


class UsuarioFraude(HttpUser):
    """Simula um usuário chamando a API de detecção de fraudes.

    wait_time: intervalo entre requisições (0.5 a 1.5 segundos).
    Isso simula um usuário real — não dispara requisições sem pausa.
    """
    wait_time = between(0.5, 1.5)

    @task(3)
    def prever_transacao_unica(self):
        """Chama o endpoint /predict (transação única)."""
        self.client.post(
            "/predict",
            json=TRANSACAO_EXEMPLO,
            headers={"Content-Type": "application/json"},
            name="/predict",
        )

    @task(1)
    def prever_lote(self):
        """Chama o endpoint /predict/batch (lote de 10 transações)."""
        self.client.post(
            "/predict/batch",
            json=LOTE_EXEMPLO,
            headers={"Content-Type": "application/json"},
            name="/predict/batch",
        )

    @task(1)
    def health_check(self):
        """Chama o endpoint /health para simular monitoramento."""
        self.client.get("/health", name="/health")