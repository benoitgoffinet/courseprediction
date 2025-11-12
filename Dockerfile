FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Laisse Azure fournir PORT ; par défaut 8051 si absent
ENV PORT=8051

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --only-binary=:all: --prefer-binary \
    "numpy==1.26.4" \
    "scikit-learn>=1.3,<1.6" \
    "shap>=0.45,<0.46"

COPY requirements.txt /app/requirements.txt
RUN pip install --prefer-binary -r /app/requirements.txt

ARG GIT_SHA=dev
RUN echo $GIT_SHA > /app/version.txt

# Copie tout le code : vérifie que .dockerignore n'exclut pas app.py !
COPY . /app

EXPOSE 8051
# Lancement : si app.py n’existe pas, tente app/app.py, sinon affiche le contenu pour debug
CMD sh -lc '\
  set -e; \
  TARGET=""; \
  if [ -f app.py ]; then TARGET=app.py; \
  elif [ -f app/app.py ]; then TARGET=app/app.py; \
  else echo "❌ app.py introuvable (ni ./app.py ni ./app/app.py). Contenu /app:"; ls -la; exit 2; fi; \
  exec streamlit run "$TARGET" --server.address=0.0.0.0 --server.port="${PORT:-8051}" --server.headless=true \
