import streamlit as st
import pandas as pd
import pickle
import subprocess
import sys
import requests
import time
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder 
import os
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pickle
import tempfile




def get_data(local_path="data/ndf.pkl"):
    """
    Récupère le dernier dataset depuis MLflow si dispo,
    sinon lit le fichier local.
    """
    ndf = None
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Datasets")

    if experiment is not None:
        try:
            # Récupère le dernier run
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=50
            )
            # Filtrer ceux dont le run_name commence par "data"
            data_runs = [r for r in runs if r.info.run_name.startswith("data")]
            if data_runs:
                run_id = data_runs[0].info.run_id
                artifact_paths = client.list_artifacts(run_id, path="data")
                
                # Cherche d'abord un CSV, sinon pickle
                csv_files = [f.path for f in artifact_paths if f.path.endswith(".csv")]
                pkl_files = [f.path for f in artifact_paths if f.path.endswith(".pkl")]

                if csv_files:
                    local_tmp = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{csv_files[0]}")
                    ndf = pd.read_csv(local_tmp)
                    print("✅ Dataset chargé depuis MLflow (CSV)")
                elif pkl_files:
                    local_tmp = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{pkl_files[0]}")
                    with open(local_tmp, "rb") as f:
                        ndf = pickle.load(f)
                    print("✅ Dataset chargé depuis MLflow (Pickle)")
        except Exception as e:
            print(f"⚠️ Impossible de récupérer depuis MLflow : {e}")

    # Fallback local si rien trouvé
    if ndf is None:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Impossible de trouver {local_path} localement")
        with open(local_path, "rb") as f:
            ndf = pickle.load(f)
        print("📂 Dataset chargé localement")

    return ndf

def log_dataframe_to_mlflow(df, run_name, artifact_name="data"):
    """
    Enregistre un DataFrame dans MLflow comme artifact CSV.
    """
    # Si aucun run n'est actif, on en démarre un
    if mlflow.active_run() is None:
        mlflow.set_experiment("Datasets")
        mlflow.start_run(run_name=run_name)
        run_started_here = True
    else:
        run_started_here = False

    # Crée un fichier temporaire
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = "data/data.csv"
        os.makedirs("data", exist_ok=True)  # au cas où le dossier n'existe pas
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="data")

    # Si on a démarré le run ici, on le termine
    if run_started_here:
        mlflow.end_run()

    print(f"✅ DataFrame loggé dans MLflow sous le nom '{artifact_name}'")



st.title("Application de suivi et prédiction de performances de course ")

mlflow.set_tracking_uri("http://127.0.0.1:5000")





if "model_train" not in st.session_state:
    st.session_state.model_train = 0

# Initialisation ou chargement du DataFrame
FILE_PATH = "data/ndf.pkl"

# --- Chargement ou initialisation ---
ndf  = get_data()


# --- Formulaire pour ajouter une nouvelle performance ---
st.subheader("Ajouter une performance – Plus vous enregistrez de courses, plus les prédictions seront précises (minimum 2)")
with st.form("add_performance"):
  athlete = st.text_input("Nom de l'athlète")
  distance = st.number_input("Distance (km)", min_value=0.0, value=5.0)
  sexe = st.selectbox("Sexe", options=['M','F'])
  st.write("Temps :")
    
    # Minutes et secondes sur la même ligne
  col1, col2 = st.columns(2)
  with col1:
        minutes = st.number_input("Minutes", min_value=0, value=30)
  with col2:
        secondes = st.number_input("Secondes", min_value=0, max_value=59, value=0)


  submitted = st.form_submit_button("Ajouter")
  if submitted and athlete:
# Ajouter au DataFrame
    
    new_row = pd.DataFrame({
     'athlete':[athlete],
     'gender':[sexe],
     'distance (m)':[distance * 1000],
     'elapsed time (s)':[minutes * 60 + secondes]
    })
    ndf = pd.concat([ndf, new_row], ignore_index=True)
    athlete_df = ndf[ndf["athlete"] == athlete]
    athlete_count = len(athlete_df)
    run_name = f"data_{athlete}_{athlete_count}"
    log_dataframe_to_mlflow(ndf, run_name)
    st.success(f"Performance ajoutée pour {athlete}")



# Affichage du DataFrame
if athlete.strip() != '':
   athlete_df = ndf[ndf["athlete"] == athlete]

# Sélectionner seulement les colonnes concernées
   cols_to_display = ["athlete", "distance (m)", "gender", "elapsed time (s)"]
   athlete_df = athlete_df[cols_to_display]
   st.subheader(f"Historique des performances de {athlete}")
   st.dataframe(athlete_df)


# --- Réentraînement du modèle si données suffisantes ---
# Filtrer le DataFrame pour l'athlète sélectionné
   athlete_df = ndf[ndf["athlete"] == athlete]


# Vérifier qu'il a au moins 2 performances
   if len(athlete_df) < 2:
    st.warning(f"{athlete} n'a pas encore assez de performances pour la prédiction.")
    st.session_state.model_train = 0
   else:
    st.write(f"{athlete} a au moins 2 performances, on peut entraîner le modèle.")
     # Bouton pour entraîner le modèle
    if st.button("Entraîner le modèle"):
     num_rows = len(ndf[ndf['athlete'] == athlete])
     run_name = f"{athlete}_{num_rows}"

# Stocker dans la session
     st.session_state.run_name = run_name
     ndf['athlete'] = ndf['athlete'].astype(str)
     dataexplicative = ndf.drop(columns=["T/K", 'timestamp', 'elapsed time (s)', 'M/S', 'K/H', 'average heart rate (bpm)', 'elevation gain (m)'])
     dataexplicative["distance (m)"] = dataexplicative["distance (m)"] / 1000# reduire impact de la variable
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
     st.session_state.best_model = best_model

     end_time = time.time()
     training_duration = end_time - start_time  

# Formatage en heures, minutes, secondes
     hours, rem = divmod(training_duration, 3600)
     minutes, seconds = divmod(rem, 60)
     duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

     mlflow.log_metric("R2_Score_VC", grid_search.best_score_)
     mlflow.log_params(grid_search.best_params_)
     mlflow.set_tag("training_duration_str", duration_str)
     mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path=st.session_state.run_name
    )
     st.success("Entraînement terminé !")
     st.session_state.model_train = 1
    
     

if st.session_state.model_train == 1:        
     with st.form("Prédire une course"):
      athlete_pred = st.text_input("Athlète", value=athlete, disabled=True)
      sexe_pred = st.selectbox("Sexe", options=['M','F'], key='sexe_pred')
      distance_pred = st.number_input("Distance (km)", min_value=0.0, value=5.0, key='dist_pred')
      predict_submitted = st.form_submit_button("Prédire le temps")


      if predict_submitted:
             X_new = pd.DataFrame({
                  "athlete": [athlete_pred],        
                  "gender": [sexe_pred],
                  "distance (m)": [distance_pred] # par besoin de mettre en 1000 car j'ai entrainé en kilometre
          })
             X_new['athlete'] = X_new['athlete'].astype(str)
             X_new['gender'] = X_new['gender'].astype(str)

             
             best_model = st.session_state.best_model
             predicted_time = best_model.predict(X_new)[0]
             minutes = int(predicted_time // 60)
             seconds = int(predicted_time % 60)
             st.success(f"Temps prédit pour {athlete_pred} : {minutes} min {seconds} sec")