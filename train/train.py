import datetime
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder 
import mlflow
import mlflow.sklearn
import time
import os

# Définir l'expérience
mlflow.set_experiment("Mon_Expérience_ML")

# Démarrer une nouvelle exécution
with mlflow.start_run(run_name="Entraînement_1"):
    # Ton code d'entraînement ici
    pass




BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # dossier contenant train.py
FILE_PATH = os.path.join(BASE_DIR, "..", "data", "ndf.pkl")


# --- Chargement ou initialisation ---
with open(FILE_PATH, "rb") as f:
            ndf = pickle.load(f)

def send_to_workflow(message):
    """
    Envoie un message au workflow externe avec timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    
    # On suppose que st.session_state.workflow_logger est une fonction qui envoie le message
    if 'workflow_logger' in st.session_state:
        st.session_state.workflow_logger(full_message)
    else:
        # Optionnel : si le workflow n'est pas défini, lever une erreur
        raise ValueError("Workflow logger non défini dans st.session_state")
        # Variables explicatives et cible
       #choix des variables explicatives



ndf['athlete'] = ndf['athlete'].astype(str)
dataexplicative = ndf.drop(columns=["T/K", 'timestamp', 'elapsed time (s)', 'M/S', 'K/H', 'average heart rate (bpm)'])
target = ndf['elapsed time (s)'] 
X = dataexplicative
y = target

# 📊 Créer des quartiles pour la stratification (les quartiles de y)
y_quartiles = pd.qcut(y, q=4, labels=[0, 1, 2, 3])  # Diviser en 4 groupes pour la stratification

# 🎲 Séparation des données avec stratification basée sur les quartiles de la cible
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42#, stratify=y_quartiles  # Stratification sur y_quartiles
 )

# 🔧 Pipeline de prétraitement

preprocessor = ColumnTransformer(
transformers=[
('num', StandardScaler(), X_train.select_dtypes(exclude='object').columns.tolist()),
('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), X_train.select_dtypes(include='object').columns.tolist())
]
)

# 💡 Pipeline complet avec GridSearchCV
pipeline = Pipeline(steps=[
('preprocessing', preprocessor),
('model', GradientBoostingRegressor(random_state=42)) 
])

# Paramètres pour GridSearchCV
param_grid = {
'model__max_depth': [5],  # 
'model__n_estimators': [300, 200],  # Add more parameters for the model if needed
'model__min_samples_split': [2, 3, 4]
    # Add other hyperparameters for preprocessing steps if needed
}

# GridSearchCV
grid_search = GridSearchCV(
pipeline,
param_grid,
scoring='r2',
cv=5,
verbose=1,
n_jobs=-1
)
start_time = time.time()
# 🚀 Entraînement avec GridSearchCV
grid_search.fit(X_train, y_train)


# ✅ Meilleur modèle trouvé
best_model = grid_search.best_estimator_
MODEL_DIR = os.path.join(os.getcwd(), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Chemin complet pour sauvegarder le modèle
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

end_time = time.time()
training_duration = end_time - start_time  

# Formatage en heures, minutes, secondes
hours, rem = divmod(training_duration, 3600)
minutes, seconds = divmod(rem, 60)
duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# Envoyer le temps d'entraînement au workflow

mlflow.log_metric("R2_Score_VC", grid_search.best_score_)
mlflow.log_params(grid_search.best_params_)
mlflow.set_tag("training_duration_str", duration_str)




# 🟢 Prédictions sur l'ensemble de test
y_pred = best_model.predict(X_test)
r2_test = r2_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)

mlflow.log_metric("R2_Score_test", r2_test)
mlflow.log_metric("MSE_test", mse_test)


# 🟢 Prédictions sur l'ensemble d'entraînement
y_pred_train = best_model.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)

mlflow.log_metric("R2_Score_train", r2_train)
mlflow.log_metric("MSE_train", mse_train)

# Enregistrer le modèle
mlflow.sklearn.log_model(best_model, "model")

