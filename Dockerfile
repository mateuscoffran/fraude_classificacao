# ══════════════════════════════════════════════════════════════════
# Stage 1 — builder
# Instala as dependências em um ambiente separado.
# Isso evita que ferramentas de build (pip, poetry, cache) entrem
# na imagem final, mantendo-a menor e mais segura.
# ══════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS builder

# Instala o Poetry
RUN pip install --no-cache-dir poetry==1.8.3

WORKDIR /app

# Copia apenas os arquivos de dependências primeiro.
# Isso aproveita o cache do Docker — se o pyproject.toml não mudar,
# o Docker reutiliza essa camada sem reinstalar tudo.
COPY pyproject.toml poetry.lock ./

# Instala as dependências em /app/.venv (sem o grupo dev)
RUN poetry config virtualenvs.in-project true && \
    poetry install --only main --no-interaction --no-ansi


# ══════════════════════════════════════════════════════════════════
# Stage 2 — runtime
# Imagem final, enxuta, apenas com o necessário para rodar a API.
# ══════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS runtime

# Boas práticas de segurança: nunca rodar como root em produção
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copia o virtualenv pronto do stage builder
COPY --from=builder /app/.venv /app/.venv

# Copia o código-fonte e o modelo treinado
COPY src/       ./src/
COPY models/    ./models/

# Ajusta permissões
RUN chown -R appuser:appgroup /app

# Troca para o usuário não-root
USER appuser

# Garante que o Python use o virtualenv do projeto
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Porta exposta pela API
EXPOSE 8000

# Comando para iniciar a API
# --host 0.0.0.0 permite conexões de fora do container
# --workers 2    usa 2 processos paralelos para atender requisições
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]