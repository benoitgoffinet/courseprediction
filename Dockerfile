FROM python:3.10-slim

# Env de base + port par défaut (Azure fournit PORT)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PORT=8051 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Seule lib native requise (wheels numpy/sklearn/shap utilisent OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Dépendances (on force shap pour éviter la panne au runtime)
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install --prefer-binary "shap>=0.45,<0.46" \
 && python -m pip install --prefer-binary -r requirements.txt \
 && python - <<'PY'
import shap, numpy, sklearn
print("✅ shap", shap.__version__, "| numpy", numpy.__version__, "| sklearn", sklearn.__version__)
PY

# Version (optionnel)
ARG GIT_SHA=dev
RUN echo $GIT_SHA > /app/version.txt

# Code
COPY . .

EXPOSE 8051

# Lance app.py ou app/app.py ; sinon affiche le contenu et quitte proprement
CMD sh -lc 'set -e; \
  if [ -f app.py ]; then TARGET=app.py; \
  elif [ -f app/app.py ]; then TARGET=app/app.py; \
  else echo "❌ app.py introuvable (./app.py ou ./app/app.py). Contenu de /app :" && ls -la && exit 2; fi; \
  exec python -m streamlit run "$TARGET" --server.address=0.0.0.0 --server.port="${PORT:-8051}" --server.headless=true'
