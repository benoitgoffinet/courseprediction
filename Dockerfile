FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Outils nécessaires (compil si jamais, git pour certaines deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# (1) Mettre pip/setuptools/wheel à jour (crucial pour wheels manylinux)
RUN python -m pip install --upgrade pip setuptools wheel

# (2) Debug: afficher les versions de wheels disponibles AVANT install
#     -> si ça échoue ici on voit clairement où ça bloque
RUN python -m pip index versions numpy || true
RUN python -m pip index versions scikit-learn || true
RUN python -m pip index versions shap || true
RUN python -m pip index versions mlflow || true
RUN python -m pip index versions azureml-mlflow || true

# (3) Pré-installer les lourds en wheels uniquement
RUN pip install --only-binary=:all: --prefer-binary \
    "numpy==1.26.4" \
    "scikit-learn>=1.3,<1.6"

# (4) Installer le reste, en verbeux pour capter l’erreur précise
COPY requirements.txt /app/requirements.txt
RUN pip install -v --prefer-binary -r /app/requirements.txt

# (5) Version & app
ARG GIT_SHA=dev
RUN echo $GIT_SHA > /app/version.txt
COPY . /app

EXPOSE 8501
CMD ["bash", "-lc", "streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT}"]
