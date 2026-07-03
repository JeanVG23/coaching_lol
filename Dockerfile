# coaching_lol — image backend FastAPI (sert aussi le frontend statique).
FROM python:3.13-slim

WORKDIR /app

# Déps d'abord (cache Docker) : poetry.lock ne bouge pas à chaque commit de code.
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry==2.3.2 \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Code de l'app + modules existants (src/). data/ est exclu (cf. .dockerignore)
# — la donnée vivra sur un volume persistant Fly, pas dans l'image.
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# uvicorn sert l'API + les fichiers statiques montés par main.py.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--app-dir", "/app/web/backend"]