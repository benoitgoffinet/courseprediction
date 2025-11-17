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
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime
from datetime import datetime
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
import shap
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import plotly.graph_objects as go
import random
import smtplib
from email.mime.text import MIMEText
import streamlit as st



def get_data(local_path, exp_name):
    """
    Récupère le dernier dataset depuis MLflow si dispo,
    sinon lit le fichier local.
    """
    ndf = None
    client = MlflowClient()
    experiment = client.get_experiment_by_name(exp_name)

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

def log_dataframe_to_mlflow(df, run_name, exp_name, artifact_name="data"):
    """
    Enregistre un DataFrame dans MLflow comme artifact CSV.
    """
    # Si aucun run n'est actif, on en démarre un
    mlflow.set_experiment(exp_name)
    if mlflow.active_run() is None:
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
    
def clean_decimal(val):
    # Si la valeur est None ou vide
        if val is None:
            return ""

        val = str(val).strip().lower().replace(",", ".")  # gère aussi les virgules
        try:
           f = float(val)
        # Si c'est un nombre entier (ex: 40.0 → "40")
           if f.is_integer():
               return str(int(f))
        # Sinon garder la valeur avec décimales
           return str(f)
        except ValueError:
        # Si ce n'est pas un nombre, on garde le texte nettoyé
            return val
            
def kmh_to_pace(v_kmh):
    sec_per_km = 3600 / v_kmh
    minutes = int(sec_per_km // 60)
    seconds = round(sec_per_km % 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return minutes, seconds
    
def submit(ndf, athlete):
    athlete_df = ndf[ndf['Name'] == athlete]
    if len(athlete_df) == 0:
      scode = st.session_state.user_code
      squestion = st.session_state.question
      sreponse = st.session_state.reponse
    else:
      scode = athlete_df["Code"].iloc[0]
      squestion = athlete_df["Question"].iloc[0]
      sreponse = athlete_df["Reponse"].iloc[0]
      
    new_row = pd.DataFrame({
     'Name':[athlete],
     'Gender':[sexe],
     'Distance':[distance * 1000],
     'Time':[minutes * 60 + secondes],
     'Code':[scode],
     'Question':[squestion],
     'Reponse':[sreponse]
    })
    ndf = pd.concat([ndf, new_row], ignore_index=True)
    if {5000, 10000, 21000, 42000} <= set(ndf.loc[ndf["Name"] == athlete, "Distance"]): # eviter que lathlete ait des perf sur toutes les distances pour éviter de biaiser la prédiction
         st.error("⚠️ Modifier la distance SVP")
    else: 
      ndf['T/K'] = (ndf['Time'] / (ndf['Distance'] / 1000))/60
      ndf['moyenne/athlete/5k'] = ndf.groupby('Name').apply(lambda g: (g['T/K'].where(g['Distance']==5000).sum() - g['T/K'].where(g['Distance']==5000)) /             (g['Distance'].eq(5000).sum() - 1)).reset_index(level=0, drop=True)
      ndf['moyenne/générale/5k'] = ndf.groupby('Gender').apply(
      lambda g: g[g['Distance'] == 5000]['T/K'].mean()
      ).reindex(ndf['Gender']).values
      ndf['Rapport(Athlete/moyenne)5k au km'] = (ndf['moyenne/athlete/5k'] - ndf['moyenne/générale/5k']) 
      ndf['moyenne/athlete/10k'] = ndf.groupby('Name').apply(lambda g: (g['T/K'].where(g['Distance']==10000).sum() - g['T/K'].where(g['Distance']==10000)) /          (g['Distance'].eq(10000).sum() - 1)).reset_index(level=0, drop=True)
      ndf['moyenne/générale/10k'] = ndf.groupby('Gender').apply(
      lambda g: g[g['Distance'] == 10000]['T/K'].mean()
      ).reindex(ndf['Gender']).values
      ndf['Rapport(Athlete/moyenne)10k au km'] = (ndf['moyenne/athlete/10k'] - ndf['moyenne/générale/10k']) 
      ndf['moyenne/athlete/21k'] = ndf.groupby('Name').apply(lambda g: (g['T/K'].where(g['Distance']==21000).sum() - g['T/K'].where(g['Distance']==21000)) /          (g['Distance'].eq(21000).sum() - 1)).reset_index(level=0, drop=True)
      ndf['moyenne/générale/21k'] = ndf.groupby('Gender').apply(
      lambda g: g[g['Distance'] == 21000]['T/K'].mean()
      ).reindex(ndf['Gender']).values
      ndf['Rapport(Athlete/moyenne)21k au km'] = (ndf['moyenne/athlete/21k'] - ndf['moyenne/générale/21k']) 
      ndf['moyenne/athlete/42k'] = ndf.groupby('Name').apply(lambda g: (g['T/K'].where(g['Distance']==42000).sum() - g['T/K'].where(g['Distance']==42000)) /          (g['Distance'].eq(42000).sum() - 1)).reset_index(level=0, drop=True)
      ndf['moyenne/générale/42k'] = ndf.groupby('Gender').apply(
      lambda g: g[g['Distance'] == 42000]['T/K'].mean()
      ).reindex(ndf['Gender']).values
      ndf['Rapport(Athlete/moyenne)42k au km'] = (ndf['moyenne/athlete/42k'] - ndf['moyenne/générale/42k']) 
      cols = [
      'Rapport(Athlete/moyenne)5k au km',
      'Rapport(Athlete/moyenne)10k au km',
      'Rapport(Athlete/moyenne)21k au km',
      'Rapport(Athlete/moyenne)42k au km'
      ]

      ndf['Score/athlete'] = ndf[cols].sum(axis=1, skipna=True)
      ndf['Score/athlete/moyenne'] = ndf.groupby('Name')['Score/athlete'].transform('mean')
      st.session_state.ndf = ndf
      nathlete_df = st.session_state.ndf[st.session_state.ndf["Name"] == athlete]
      athlete = st.session_state.athlete
      athlete_df = ndf[ndf["Name"] == athlete]
      athlete_count = len(athlete_df)
      run_name = f"data_{athlete}_{athlete_count}"
      log_dataframe_to_mlflow(ndf, run_name, exp_name)
# Sélection dynamique de la bonne colonne en fonction de la distance
def calculer_coef(row):
    col_moyenne = col_map[row["Distance"]]
    return row["T/K"] - ndfperfathlete[col_moyenne].iloc[0]  # ou moyenne selon ton ndf
def predict_time(model, score, gender, distance_km):
    """
    Utilitaire : prédire le temps (en minutes) pour une distance (en km)
    avec ta structure d'inputs actuelle.
    """
    X = pd.DataFrame({
        "Score/athlete": [score],
        "Distance": [np.log(distance_km * 1000)],  # distance en mètres, puis log
        "Gender": [gender]
    })
    return float(model.predict(X)[0])


def find_distance_bisection(
    model,
    score,
    gender,
    target_time=6.0,
    d_min=0.1,      # km (100 m)
    d_max=20.0,     # km
    max_iter=40,
    tol=0.01        # tolérance sur le temps (minutes)
):
    """
    Méthode de bissection pour trouver la distance (en km) telle que
    le modèle prédit target_time (en minutes).
    On suppose que le temps augmente avec la distance (fonction monotone).
    """

    # f(d) = temps(d) - target
    def f(d):
        return predict_time(model, score, gender, d) - target_time

    left = d_min
    right = d_max
    f_left = f(left)
    f_right = f(right)

    # 1) Vérifier qu'on encadre bien la cible (changement de signe)
    #    Si ce n'est pas le cas, on essaie d'élargir un peu la borne droite
    tries = 0
    while f_left * f_right > 0 and tries < 5:
        right *= 2
        if right > 100:   # sécurité
            break
        f_right = f(right)
        tries += 1

    # Toujours pas de changement de signe -> on abandonne proprement
    if f_left * f_right > 0:
        return None, None

    best_distance = None
    best_error = float("inf")

    # 2) Bissection
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = f(mid)
        t_mid = f_mid + target_time  # puisqu'on a f_mid = temps - target
        error = abs(t_mid - target_time)

        # On garde la meilleure solution vue
        if error < best_error:
            best_error = error
            best_distance = mid

        # Condition d'arrêt sur la précision
        if error < tol:
            break

        # On choisit le sous-intervalle qui contient le changement de signe
        if f_left * f_mid <= 0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid
    return best_distance

def next_weekday(start_date, weekday=0):
    # weekday: 0=lundi, 1=mardi, …, 6=dimanche
    days_ahead = weekday - start_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return start_date + timedelta(days=days_ahead)


# connexion à mlflow soit en ligne soit local
AZURE_URI = (
    "azureml://canadacentral.api.azureml.ms/mlflow/v1.0/"
    "subscriptions/b115f392-8b15-499a-a548-edd84815dbcb/"
    "resourceGroups/predictioncourse_group/"
    "providers/Microsoft.MachineLearningServices/"
    "workspaces/courseapied-ws"
)

# Par défaut, on utilise le serveur local (utile pour le dev)
USE_AZURE = os.getenv("USE_AZURE_MLFLOW") == "1"

if USE_AZURE:
    print("🟦 MLflow connecté à Azure ML")
    mlflow.set_tracking_uri(AZURE_URI)
else:
    print("🟩 MLflow connecté au serveur local (127.0.0.1:5000)")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    edd84815dbcb/resourceGroups/predictioncourse_group/providers/Microsoft.MachineLearningServices/workspaces/courseapied-ws")
    
# Nom de la nouvelle expérience

exp_name = "CourseAPied_Experience"
exp_nameid = "CourseAPied_Experienceid"
# Crée l'expérience si elle n'existe pas, sinon récupère-la
mlflow.set_experiment(exp_name)


if "page" not in st.session_state:
    st.session_state.page = "application"

if "athlete" not in st.session_state:
    st.session_state.athlete = " "

if "user_code" not in st.session_state:
    st.session_state.user_code = " "

if "question" not in st.session_state:
    st.session_state.question = " "

if "reponse" not in st.session_state:
    st.session_state.reponse = " "

if "model_train" not in st.session_state:
    st.session_state.model_train = 0

if "statistique" not in st.session_state:
    st.session_state.statistique = 0

if "supprimer" not in st.session_state:
    st.session_state.supprimer = 0

if "admis" not in st.session_state:
    st.session_state.admis = 0
    
if "identification_echec" not in st.session_state:
    st.session_state.identification_echec = 0

if "ndf" not in st.session_state:
    st.session_state.ndf = get_data("../data/ndfs.pkl", exp_name)

if "idf" not in st.session_state:
    st.session_state.identification_idf = get_data("../data/idf.pkl", exp_nameid)

if "suite" not in st.session_state:
    st.session_state.suite = 0

if "identification" not in st.session_state:
    st.session_state.identification = 0

if "distance" not in st.session_state:
    st.session_state.distance = 0

if "clicsup" not in st.session_state:
    st.session_state.clicsup = 0

if "prediction" not in st.session_state:
    st.session_state.prediction = 0
    
if "score_athlete" not in st.session_state:
    st.session_state.score_athlete = 0



# Initialisation ou chargement du DataFrame
FILE_PATH = "../data/ndfs.pkl"

# --- Chargement ou initialisation ---
ndf  = st.session_state.ndf
idf = st.session_state.identification_idf 


if st.session_state.identification == 0:
    with st.form("identificationuser"):
        st.session_state.athlete = st.text_input(
    "Votre nom d'utilisateur",
    value=st.session_state.athlete
)
        
        
        athlete = st.session_state.athlete
        athlete_df = idf[idf['Name'] == athlete]
        if (len(athlete_df)) == 0 and athlete != " " : 
            if st.session_state.user_code == " ":
              nombrealeatoire = random.randint(0, 9999)
              st.session_state.user_code = athlete + str(nombrealeatoire)
              user_code = st.session_state.user_code
              st.info(f"🔑 Code personnel associé : `{user_code}`")
              st.warning("Attention ! Vous aurez besoin de ce code pour vous connecter à votre profil")
              st.success("Répondez à la question, pour retrouver votre code en cas de perte")
            if st.session_state.user_code != " ":
              st.session_state.question = st.selectbox(
            "Question de vérification :",
            options=[
            "Quel est le prénom de ton premier coach ?",
            "Dans quelle ville as-tu couru ta première course ?",
            "Quel est ton chiffre porte-bonheur ?",
            "Quel est le nom de ton club de course ?",
            "Quelle marque de chaussures utilises-tu le plus souvent ?",
            "Quel est ton record personnel sur 10 km (approximatif) ?",
            "Quel est ton plat préféré après une course ?",
            ]
            )
              question = st.session_state.question
              st.session_state.reponse = st.text_input("Réponse")
              reponse = st.session_state.reponse
              if reponse:
                 st.session_state.admis = 1
                 row = {
    "Name": st.session_state.athlete,
    "Code": st.session_state.user_code,
    "Question": st.session_state.question,
    "Reponse": st.session_state.reponse,
    "nombreconnexion": 1,
    "datepremiereconnexion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "datederniereconnexion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "echecconnexion": 0,
    "perfenregistre": 0,
    "nombredeprediction": 0,
    "messageenvoye" : 0,
    "messagesansreponse" : 0
                     
}
                 idf = pd.concat([idf, pd.DataFrame([row])], ignore_index=True) 
                 nrun_name = f"data_{athlete}_{1}"
                 log_dataframe_to_mlflow(idf, nrun_name, exp_nameid)
                 st.session_state.identification_idf = idf
              else:
                 st.warning("Répondez à la question")
        if (len(athlete_df)) == 1 :
            nuser_code = st.text_input("Code")
            if nuser_code != "":
              nuser_code = clean_decimal(nuser_code)
              basecode = clean_decimal(athlete_df.iloc[0]['Code'])
              if nuser_code == basecode:
                   st.success(f"Identification réussi")
                   st.session_state.admis = 1
                   st.session_state.identification_echec = 0
                   idf.loc[idf["Name"] == athlete, "nombreconnexion"] = (
                   idf.loc[idf["Name"] == athlete, "nombreconnexion"] + 1
                   )
                   idf.loc[idf["Name"] == athlete, "datederniereconnexion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                   nrun_name = f"data_{athlete}_{1}"
                   log_dataframe_to_mlflow(idf, nrun_name, exp_nameid)
                   st.session_state.identification_idf = idf
                   
              else:  
                    base = athlete_df.iloc[0]['Code']
                    st.error(f"Nous n'arrivons pas à vous identifier")
                    st.session_state.identification_echec = 1
                    st.session_state.admis = 0
                    idf["echecconnexion"] = idf["echecconnexion"].fillna(0)
                    idf.loc[idf["Name"] == athlete, "echecconnexion"] = (
                    idf.loc[idf["Name"] == athlete, "echecconnexion"] + 1
                    )
                    nrun_name = f"data_{athlete}_{1}"
                    log_dataframe_to_mlflow(idf, nrun_name, exp_nameid)
                    st.session_state.identification_idf = idf
              if st.session_state.identification_echec == 1:
                    st.warning("Vous avez oublié votre code? Répondez à la question")
                    nquestion = st.selectbox(
                    "Question de vérification :",
                    options=[
                    "Quel est le prénom de ton premier coach ?",
                    "Dans quelle ville as-tu couru ta première course ?",
                    "Quel est ton chiffre porte-bonheur ?",
                    "Quel est le nom de ton club de course ?",
                    "Quelle marque de chaussures utilises-tu le plus souvent ?",
                    "Quel est ton record personnel sur 10 km (approximatif) ?",
                    "Quel est ton plat préféré après une course ?",
                    ]
                    )
      
                    nreponse = st.text_input("Réponse")     
                    reponsebase = clean_decimal(athlete_df.iloc[0]["Reponse"])
                    if (
                    nquestion == athlete_df.iloc[0]["Question"]
                    and nreponse == reponsebase
                      ):
                          st.success(f"Voici le code d'accès pour {athlete} : {basecode}")
                          st.session_state.identification_echec == 0
                    else:
                          basequestion = clean_decimal(athlete_df.iloc[0]["Question"])
                          basereponse = athlete_df.iloc[0]["Reponse"]
                          st.session_state.identification_echec == 1    
        submitted = st.form_submit_button("OK") 
        if submitted and st.session_state.admis == 1:
           st.session_state.identification = 1
           st.session_state.page = "application"
           st.rerun()


if st.session_state.identification == 1:
 st.sidebar.title("Navigation")

 if st.sidebar.button("Déconnexion") :
        st.session_state.identification = 0
        st.session_state.athlete = " "
        st.session_state.user_code = " "
        st.session_state.question = " "
        st.session_state.reponse = " "
        st.session_state.admis = 0
        st.session_state.page = "Déconnexion"
        st.rerun()
     
 if st.sidebar.button("🔑 Admin") and st.session_state.athlete.strip() == "Benoitgofadmin" :
        st.session_state.page = "Admin"
 
 if st.sidebar.button("🏃 Prédictions"):
    st.session_state.page = "application"

 if st.sidebar.button("📘 Programme"):
    st.session_state.page = "Programme"
     
 
 if st.sidebar.button("📊 Statistiques"):
    st.session_state.page = "Statistiques"


 if st.sidebar.button("📞 Contact"):
    st.session_state.page = "contact"

# Page admin
 if st.session_state.page == "Admin": 
     st.title("🔑 Admin ")
     idf['nombreconnexion'] = idf['nombreconnexion'].fillna(0)
     noadminidf = idf[idf["Name"] != "Benoitgofadmin"]
     connexiontotal = noadminidf['nombreconnexion'].sum()
     st.write(f"Nombre d'utilisateur = {len(noadminidf)} |  Nombre de connexion = {connexiontotal}")
     # entrainement du modele

  
     if st.button("Entrainement du modèle"): 
      athlete = st.session_state.athlete  
      num_rows = len(ndf[ndf['Name'] == athlete])
      run_name = f"{athlete}_{num_rows}-pipeline"

# Stocker dans la session
      st.session_state.run_name = run_name
      ndf['Name'] = ndf['Name'].astype(str)
      dataexplicative = ndf[['Distance', 'Score/athlete', 'Gender']]
      dataexplicative['Distance'] = np.log(dataexplicative['Distance'])
      target = np.log(ndf['Time']) 
      X = dataexplicative
      y = target



      X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
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
      ('model', ElasticNet(random_state=42)) 
      ])
      
      param_grid = {
     'model__alpha': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
     'model__l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1] 
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



      pipeline = best_model  # rename for clarity


# Log dans MLflow
      ndfperfathlete = ndf[ndf['Name'] == athlete]
      mlflow.set_experiment("pipeline")
      with mlflow.start_run(run_name=st.session_state.run_name):
       with tempfile.TemporaryDirectory() as tmpdir:
        path_model = os.path.join(tmpdir, "model_pipeline.pkl")

        # Sauvegarder ton pipeline avec pickle
        with open(path_model, "wb") as f:
            pickle.dump(pipeline, f)

        # Log du modèle comme artefact (comme ton CSV)
        mlflow.log_artifact(path_model, artifact_path="model_pipeline")

    # --- Autres logs MLflow ---
       mlflow.log_param("model_type", "ElasticNet")
       mlflow.log_metric("R2_Score_VC", grid_search.best_score_)
       mlflow.log_params(grid_search.best_params_)
       mlflow.set_tag("training_duration_str", duration_str)

    # --- Log des performances de l'athlète ---
       with tempfile.TemporaryDirectory() as tmpdir:
        path_csv = os.path.join(tmpdir, f"{athlete}_performances.csv")
        ndfperfathlete.to_csv(path_csv, index=False)
        mlflow.log_artifact(path_csv, artifact_path="athlete_data")

      st.session_state.pipeline = pipeline
      st.success(f"Modèle entrainé")
      st.session_state.model_train = 1
     st.write("📬 Messages reçus (MLflow)")
     mlflow.end_run() 
# Nom de ton expérience de messages
     exp_name = "Message"

# Récupérer l'expérience
     client = MlflowClient()
     exp = client.get_experiment_by_name(exp_name)
     
     if exp is None:
      st.warning("Aucune expérience 'Messages_Utilisateurs' trouvée.")
     else:
    # Récupérer tous les runs (messages)
      mlflow.set_experiment(exp_name)
      runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["attributes.start_time DESC"])
      total_messages = len(runs)
      messages_sans_reponse = sum(
      1 for run in runs if run.data.params.get("reponse", "❌ Pas encore de réponse") == "❌ Pas encore de réponse"
      )
      st.write(f"Nombre de message = {total_messages} |  Message sans réponse = {messages_sans_reponse}")   
      if not runs:
        st.info("Aucun message enregistré pour le moment.")
      else:
        
        for run in runs:
            params = run.data.params

            utilisateur = params.get("utilisateur", "Inconnu")
            message = params.get("message", "—")
            reponse = params.get("reponse", "❌ Pas encore de réponse")
            date = params.get("date", "—")

            with st.expander(f"💬 Message de {utilisateur} — {date}"):
                st.write(f"**Message :** {message}")
                st.write(f"**Réponse :** {reponse}")
                # Zone de réponse
                nouvelle_reponse = st.text_area(
                    "Votre réponse :", 
                    key=f"reponse_{run.info.run_id}", 
                    height=100
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Envoyer la réponse à {utilisateur}", key=f"btn_{run.info.run_id}"):
                        if nouvelle_reponse.strip():
                            mlflow.end_run()
                            with mlflow.start_run(run_id=run.info.run_id):
                                mlflow.log_param("reponse", nouvelle_reponse)
                                mlflow.log_param("date_reponse", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            st.success("✅ Réponse enregistrée dans MLflow !")

                with col2:
                    # 🗑️ Bouton de suppression
                    if st.button(f"Supprimer le message", key=f"delete_{run.info.run_id}"):
                        client.delete_run(run.info.run_id)
                        st.warning(f"🗑️ Message de {utilisateur} supprimé avec succès.")
                        st.rerun()  # recharge la page


     
# --- Formulaire pour ajouter une nouvelle performance ---
 if st.session_state.page == "application":
  st.title("Suivi et prédiction de performances de course ")
  st.subheader("Ajouter une performance – Minimum 2 sur la meme distance")
  with st.form("add_performance"):
   st.text_input(
    "Nom de l'athlète",
    key="athlete",
    value=st.session_state.athlete
)
   athlete = st.session_state.athlete
   st.session_state.distance = st.selectbox(
    "Choisis la distance",
    options=[10, 21, 42],
    format_func=lambda x: f"{x} km"
)
   distance = st.session_state.distance
   sexe = st.selectbox("Sexe", options=['M','F'])
   st.write("Temps :")
    
    # Minutes et secondes sur la même ligne
   col1, col2 = st.columns(2)
   with col1:
        minutes = st.number_input("Minutes", min_value=0, value=60)
   with col2:
        secondes = st.number_input("Secondes", min_value=0, max_value=59, value=0)

   athlete_df = ndf[ndf['Name'] == athlete]
   if athlete != ' ':
    athlete_df = ndf[ndf['Name'] == athlete]
   
    
      
      
   submitted = st.form_submit_button("Ajouter")
   athlete_df = ndf[ndf["Name"] == athlete]
   if not athlete_df.empty:
    if athlete_df.iloc[0]["Gender"] != sexe:
      st.warning(f"Le sexe ne correspond pas")
      st.session_state.admis =0

  
   if submitted and athlete and st.session_state.admis == 1:
     submit(ndf, athlete)         
     st.success(f"Performance ajoutée pour {athlete}")
     st.session_state.model_train = 0
     idf = st.session_state.identification_idf 
     idf["perfenregistre"] = idf["perfenregistre"].fillna(0)
     idf.loc[idf["Name"] == athlete, "perfenregistre"] = (
     idf.loc[idf["Name"] == athlete, "perfenregistre"] + 1
     )
     nrun_name = f"data_{athlete}_{1}"
     log_dataframe_to_mlflow(idf, nrun_name, exp_nameid)
     st.session_state.identification_idf = idf

#entrainement model(a retirer quand suffisamment de data avec variable datascoremoyenne) = laisser seulement lentrainement dans admin
     ndf = st.session_state.ndf 
     athlete = st.session_state.athlete  
     num_rows = len(ndf[ndf['Name'] == athlete])
     run_name = f"{athlete}_{num_rows}-pipeline"

# Stocker dans la session
     st.session_state.run_name = run_name
     ndf['Name'] = ndf['Name'].astype(str)
     dataexplicative = ndf[['Distance', 'Score/athlete', 'Gender']]
     dataexplicative['Distance'] = np.log(dataexplicative['Distance'])
     target = np.log(ndf['Time']) 
     X = dataexplicative
     y = target



     X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42
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
      ('model', ElasticNet(random_state=42)) 
      ])
      
     param_grid = {
     'model__alpha': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
     'model__l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1] 
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



     pipeline = best_model  # rename for clarity


# Log dans MLflow
     ndfperfathlete = ndf[ndf['Name'] == athlete]
     mlflow.set_experiment("pipeline")
     with mlflow.start_run(run_name=st.session_state.run_name):
       with tempfile.TemporaryDirectory() as tmpdir:
        path_model = os.path.join(tmpdir, "model_pipeline.pkl")

        # Sauvegarder ton pipeline avec pickle
        with open(path_model, "wb") as f:
            pickle.dump(pipeline, f)

        # Log du modèle comme artefact (comme ton CSV)
        mlflow.log_artifact(path_model, artifact_path="model_pipeline")

    # --- Autres logs MLflow ---
       mlflow.log_param("model_type", "ElasticNet")
       mlflow.log_metric("R2_Score_VC", grid_search.best_score_)
       mlflow.log_params(grid_search.best_params_)
       mlflow.set_tag("training_duration_str", duration_str)

    # --- Log des performances de l'athlète ---
       with tempfile.TemporaryDirectory() as tmpdir:
        path_csv = os.path.join(tmpdir, f"{athlete}_performances.csv")
        ndfperfathlete.to_csv(path_csv, index=False)
        mlflow.log_artifact(path_csv, artifact_path="athlete_data")

     st.session_state.pipeline = pipeline
     st.rerun()
     

# suppression
  
  athlete = st.session_state.athlete 
  athlete_dfn = ndf[ndf["Name"] == athlete]
  st.session_state.supprimer = 0
  if len(athlete_dfn) > 0:
    athlete_dfn = ndf[ndf["Name"] == athlete]
    if st.button("Supprimer"):
       if st.session_state.clicsup == 1:
           st.session_state.supprimer = 0
           st.session_state.clicsup = 0
       else:
           st.session_state.supprimer = 1
           st.session_state.clicsup = 1
           
    if st.session_state.supprimer == 1:
       st.subheader(f"Lignes de {athlete}")
       athlete_dfn["label"] = (
         athlete_dfn["Name"].astype(str) + " | " +
         athlete_dfn["Gender"].astype(str) + " | " +
         athlete_dfn["Distance"].astype(str) + " | " +
         (athlete_dfn["Time"] / 60).round(2).astype(str) + " min"
)

       # --- Permettre de cocher les lignes à supprimer ---
       indices_to_delete = st.multiselect(
        "Sélectionnez les lignes à supprimer (par Distance) :",
         athlete_dfn["label"].astype(str),
         placeholder="🔍 Recherchez ou sélectionnez des distances..."
)
       if st.button("🗑️ Supprimer les lignes sélectionnées"):
    # Identifier les index à supprimer
           idx_to_drop = athlete_dfn[athlete_dfn["label"].isin(indices_to_delete)].index

    # Supprimer dans le DataFrame principal
           ndf = ndf.drop(idx_to_drop)
           st.session_state.ndf = ndf
           st.success(f"{len(idx_to_drop)} ligne(s) supprimée(s) pour {athlete}")
           athlete_df = ndf[ndf["Name"] == athlete]
           athlete_count = len(athlete_df)
           run_name = f"data_{athlete}_{athlete_count}"
           log_dataframe_to_mlflow(ndf, run_name, exp_name)
           st.session_state.supprimer = 0
  distance = st.session_state.distance  
  max_count = athlete_df['Distance'].value_counts().max()
  athlete_ndf = ndf[ ndf["Name"] == st.session_state.athlete]
  if athlete_ndf.empty:
      score_athlete = 0
  else:
      score_athlete = athlete_ndf['Score/athlete/moyenne'].iloc[0]
  if score_athlete == 0:
     st.warning(f"{athlete} n'a pas encore assez de performances pour faire des prédictions. (au moins 2 sur la meme distance)")
     st.session_state.model_train = 0
  else:
     df_une_ligne = athlete_df[athlete_df.groupby('Distance')['Distance'].transform('count') == 1]
     for _, row in df_une_ligne.iterrows():
        st.warning(f"⚠️  Attention : il n'y a qu'une seule performance pour la distance {(row['Distance']/1000)}KM .il en faut 2 pour que cette distance soit comptabilisée")
     st.write(f"{athlete} a au moins 2 performances sur une même distance, on peut faire des prédictions.")
     st.session_state.model_train = 1


  if st.session_state.model_train == 1:  
     toutes_les_durees = [10000, 21000, 42000]
     durees_faites = ndf.loc[ndf["Name"] == athlete, "Distance"].unique().tolist()
     option = [d for d in toutes_les_durees if d not in durees_faites]
     option = [d / 1000 for d in option]
     with st.form("Prédire une course"):
      athlete_pred = st.text_input("Athlète", value=athlete, disabled=True, key='athlete_pred')
      sexe_pred = st.text_input("Sexe", value=sexe, disabled=True, key='sexe_pred')
      distance_pred = st.selectbox("Choix de la distance", options=option)
      predict_submitted = st.form_submit_button("Prédire le temps")

      if not option:
          st.warning("✅ L'athlète a déjà couru toutes les distances disponibles.")
      else:
         if predict_submitted:
             # 1) Récupérer le dernier run (adapter le nom d'expérience si besoin)
             runs = mlflow.search_runs(
             experiment_names=["pipeline"], order_by=["start_time desc"], max_results=1
             )
             run_id = runs.iloc[0]["run_id"]

             # 2) Télécharger l'artefact 'model_pipeline/model_pipeline.pkl'
             client = MlflowClient()
             local_dir = client.download_artifacts(run_id, "model_pipeline", dst_path="./restore")
             pkl_path = os.path.join(local_dir, "model_pipeline.pkl")

             # 3) Charger la pipeline
             pipeline = joblib.load(pkl_path)   # ou: pickle.load(open(pkl_path, "rb"))
             score_athlete = ndf.loc[ndf['Name'] == athlete_pred, 'Score/athlete/moyenne'].iloc[0]
             X_new = pd.DataFrame({
                  "Score/athlete": score_athlete,        
                  "Distance": np.log([distance_pred * 1000]),
                  "Gender": sexe_pred
          })
             
             predicted_time = pipeline.predict(X_new)[0]
             predicted_time = np.exp(predicted_time)
             minutes = int(predicted_time // 60)
             seconds = int(predicted_time % 60)
             st.success(f"Temps prédit pour {athlete_pred} : {minutes} min {seconds} sec")
             idf = st.session_state.identification_idf 
             idf["nombredeprediction"] = idf["nombredeprediction"].fillna(0)
             idf.loc[idf["Name"] == athlete, "nombredeprediction"] = (
             idf.loc[idf["Name"] == athlete, "nombredeprediction"] + 1
             )
             nrun_name = f"data_{athlete}_{1}"
             log_dataframe_to_mlflow(idf, nrun_name, exp_nameid)
             st.session_state.identification_idf = idf


if st.session_state.page == "Statistiques":

         
   st.title("📊 Statistiques")
   ndfstate = ndf.copy()
   ndfstate['km/h'] = 60 / ndfstate['T/K']
   ndfstate['Time/minute'] = ndfstate['Time'] /60
   
 
   st.markdown("🏆 Top 10 par distance")
   
# --- Sélection de la distance ---
   distances = sorted(ndfstate["Distance"].unique())
   sexe = sorted(ndfstate["Gender"].unique())
   col1, col2 = st.columns(2)

   with col1:
        choix_distance = st.selectbox("🏁 Choisissez une distance :", distances)

   with col2:
        choix_sexe = st.selectbox("🏁 Choisissez un sexe :", sexe)
       
   fndfstate = ndfstate[['Name', 'Gender', 'km/h', 'Time/minute', 'Distance']]
# --- Filtrage et tri ---
   top10 = (
     fndfstate[
        (fndfstate["Distance"] == choix_distance) &
        (fndfstate["Gender"] == choix_sexe)
    ]
     .sort_values(by="km/h", ascending=False)
     .head(10)
     .reset_index(drop=True)
)

# --- Classement + médailles ---
   top10.insert(0, "Rang", range(1, len(top10) + 1))
   if len(top10) > 0:
      top10.loc[0, "Rang"] = "🥇"
   if len(top10) > 1:
      top10.loc[1, "Rang"] = "🥈"
   if len(top10) > 2:
      top10.loc[2, "Rang"] = "🥉"

# --- Affichage ---
   st.subheader(f"📊 Top 10 – {choix_distance}{choix_sexe}")
   st.dataframe(top10.style.format({"km/h": "{:.2f}", "Time/minute": "{:.2f}"}), use_container_width=True, hide_index=True) 
   athlete = st.session_state.athlete
   
   
   st.markdown("### Vitesse générale sur les différentes distances") 
   athlete_df = ndf[ndf["Name"] == athlete].copy()

  
   distances = [10, 21, 42]

# Colonnes correspondantes

   moy_cols = [f"moyenne/générale/{d}k" for d in distances]


   moyenne_tk = [
    athlete_df[c].mean()
    for c in moy_cols
    if c in athlete_df.columns and not athlete_df[c].isna().all()
]
   
   vitesse_moyenne = [60 / x if not np.isnan(x) and x > 0 else np.nan for x in moyenne_tk]
# Créer le graphique
   fig = go.Figure()



# Courbe moyenne
   fig.add_trace(go.Scatter(
    x=distances,
    y=vitesse_moyenne,
    mode='lines+markers',
    name="Moyenne générale",
    line=dict(color='orange', width=3, dash='dash'),
    customdata=np.array(vitesse_moyenne).reshape(-1, 1),
    hovertemplate=(
        "<b>%{x} km</b><br>"
        "Vitesse : %{customdata[0]:.2f} km/h<extra></extra>"
    )
))

# Mise en forme
   fig.update_layout(
    xaxis_title="Distance (km)",
    yaxis_title="Allure (Km/h)",
    template="plotly_white",
    yaxis=dict(autorange=True),  
    xaxis=dict(tickmode='array', tickvals=distances),
    legend=dict(
        orientation="h",      # "h" = horizontal
        yanchor="bottom",     # ancrage vertical
        y=1.02,               # place au-dessus du graphique
        xanchor="center",     # centrer horizontalement
        x=0.5
    ),
    legend_title="Moyenne générale"
)

   st.plotly_chart(fig, use_container_width=True)

# je vais maintenant rechercher la meilleur et moins bonne course de l'atlete pour chaque distance. Je ne vais pas pouvoir calculer le coef utilisé au préalable car il est calculé sur les autre course de lathlete sur meme distance pour éviter le data linkage
   ndfperf = ndf.copy()
   ndfperfathlete = ndfperf[ndfperf['Name'] == athlete]
   col_map = {
    5000: "moyenne/générale/10k",
    10000: "moyenne/générale/10k",
    21000: "moyenne/générale/21k",
    42000: "moyenne/générale/42k"
}
  
   if(len(ndfperfathlete)) > 0:
     ndfperfathlete["coef"] = ndfperfathlete.apply(calculer_coef, axis=1) # pas besoin de filtrer par sexe. ça a deja été fait au préalable
     ndfperfathlete['km/h'] = (60 / ndfperfathlete['T/K']).round(2)
     ndfperfathlete['Time/minute'] = ndfperfathlete['Time'] /60
     ndfperfathlete = ndfperfathlete[['Distance', 'km/h', 'Time/minute', "coef"]]
     ndfperfathlete["km/h"] = ndfperfathlete["km/h"].round(2)
     ndfperfathlete["Time/minute"] = ndfperfathlete["Time/minute"].round(2)
     ndfperfathlete["Distance"] = ndfperfathlete["Distance"].astype(int)

     st.markdown(f"### 📊 Meilleures performances de {athlete}")
   


# On groupe par distance
     grouped = ndfperfathlete.groupby("Distance")

# On récupère meilleure (coef min) et moins bonne (coef max)
     best_by_dist = grouped.apply(lambda x: x.loc[x["coef"].idxmin()])
     worst_by_dist = grouped.apply(lambda x: x.loc[x["coef"].idxmax()])

# On réinitialise les index pour l’affichage
     best_by_dist = best_by_dist.reset_index(drop=True)
     worst_by_dist = worst_by_dist.reset_index(drop=True)
       
   col1, col2 = st.columns(2)
   with col1:
    st.markdown("### 🥇 Meilleures performances par distance")
    if(len(ndfperfathlete)) > 0:
     best_by_dist = pd.DataFrame(best_by_dist.to_dict())
     best_by_dist["Distance"] = best_by_dist["Distance"].astype(int)
     best_by_dist["km/h"] = best_by_dist["km/h"].round(2)
     best_by_dist["Time/minute"] = best_by_dist["Time/minute"].round(2)
     best_by_dist["coef"] = best_by_dist["coef"].round(3)
     st.dataframe(best_by_dist, use_container_width=True, hide_index=True)
    else:
     st.markdown("Aucune perf")   
   with col2:
    st.markdown(f"### 🏆 Meilleur performance de {athlete}")
    if(len(ndfperfathlete)) > 0:
     best_all = ndfperfathlete.loc[ndfperfathlete["coef"].idxmin()]
     st.dataframe(best_all.to_frame().T.style.format({"coef": "{:.3f}", "Distance": "{:.0f}", "km/h": "{:.2f}", "Time/minute": "{:.2f}"}),        use_container_width=True, hide_index=True)
    else:
     st.markdown("Aucune perf")      
    

 if st.session_state.page == "Programme":
   st.title("📘 Programme") 
   athlete = st.session_state.athlete 
   athlete_ndf = ndf[ ndf["Name"] == athlete]
   if athlete_ndf.empty:
      score_athlete = 0
   else:
      score_athlete = athlete_ndf['Score/athlete/moyenne'].iloc[0]
      sexe_pred = athlete_ndf['Gender'].iloc[0]
   
   if score_athlete == 0:
       st.warning(F"Vous n'avez pas enregistré suffisamment de performance pour faire un programme")
   else:    
    exp_nameid = 'Programme'
    with st.form("programme"):
               athlete_prog = st.text_input("Athlète", value= athlete, disabled=True, key='athlete_pred')
               nombresemmaine = st.number_input(
                 "Nombre de semaines du programme",
                  min_value=4,
                  max_value=12,
                  step=1,
                  value=4,
                  format="%d"      # valeur par défaut
                )

               nombreentrainement = st.radio(
                 "Nombre d'entraînements par semaine",
                 [3, 4, 5, 6]
                  )
               nombrekilos = st.number_input(
                   "Nombre de kilomètres par semaine",
                   step=1,
                  value=30,
                 format="%d"
                   )

               option = [10, 21, 42]
               distance_pred = st.selectbox("Distance cible", options=option)
               niveau = st.selectbox("Difficulté", options=['Facile','Intermédiaire','Difficile'])
               programme_submitted = st.form_submit_button("Programme d'entrainement personalisé")   
               error = 0
                
               if nombreentrainement == 3: 
                   if nombrekilos > 40:
                       error = 1
                       message =  f"{nombrekilos} km répartis sur seulement 3 séances peut provoquer des blessures ! Réduisez à 40 km !"
                   if nombrekilos < 18:
                       error = 1
                       message =  f"Aller un petit effort ^^ . Au moins 18 km"
               if nombreentrainement == 4: 
                   if nombrekilos > 50:
                       error = 1
                       message =  f"{nombrekilos} km répartis sur seulement 4 séances peut provoquer des blessures ! Réduisez à 50 km !"
                   if nombrekilos < 25:
                       error = 1
                       message =  f"Aller un petit effort ^^ . Au moins 25 km"
               if nombreentrainement == 5:
                   if nombrekilos > 60:
                       error = 1
                       message =  f"{nombrekilos} km répartis sur seulement 5 séances peut provoquer des blessures ! Réduisez à 60 km !"
                   if nombrekilos < 30:
                       error = 1
                       message =  f"Aller un petit effort ^^ . Au moins 30 km"
               if nombreentrainement == 6:
                   if nombrekilos > 70:
                       error = 1
                       message =  f"{nombrekilos} km répartis sur seulement 6 séances peut provoquer des blessures ! Réduisez à 70 km !"
                   if nombrekilos < 40:
                       error = 1
                       message =  f"Aller un petit effort ^^ . Au moins 40 km"
               if error == 1:
                             st.warning(message)
                   
             
               if programme_submitted and error == 0:
                    df_athlete = ndf[ndf["Name"] == athlete_prog]
    
                    #eval profil(endurance ou vitesse)
                    colonnes = [
    "Rapport(Athlete/moyenne)5k au km",
    "Rapport(Athlete/moyenne)10k au km",
    "Rapport(Athlete/moyenne)21k au km",
    "Rapport(Athlete/moyenne)42k au km"
]
                    colonnes_trouvees = df_athlete[colonnes].notnull().any()
                    colonnes_trouvees = colonnes_trouvees[colonnes_trouvees].index.tolist()
                    
                    if len(colonnes_trouvees) == 1:
                        if colonnes_trouvees == "Rapport(Athlete/moyenne/5k)" or colonnes_trouvees == "Rapport(Athlete/moyenne/10k)":
                             profil = 'vitesse'
                        else:
                             profil = 'endurance'
                    else:
                         moyennes_par_colonne = {}
                         for col in colonnes_trouvees:
                              moyennes_par_colonne[col] = df_athlete[col].mean()
                         col_min = min(moyennes_par_colonne, key=moyennes_par_colonne.get)
                         ordre_colonnes = {
    "Rapport(Athlete/moyenne)5k au km": 1,
    "Rapport(Athlete/moyenne)10k au km": 2,
    "Rapport(Athlete/moyenne)21k au km": 3,
    "Rapport(Athlete/moyenne)42k au km": 4,
}
                         col_plus_basse_semantique = min(colonnes_trouvees, key=lambda c: ordre_colonnes[c])
                         col_plus_haute_semantique = max(colonnes_trouvees, key=lambda c: ordre_colonnes[c])
                         if col_min == col_plus_basse_semantique:
                             profil = 'vitesse'
                         elif col_min == col_plus_haute_semantique:
                             profil = 'endurance'
                         else:
                             if col_min == "Rapport(Athlete/moyenne/5k)" or col_min == "Rapport(Athlete/moyenne/10k)" :
                                 profil = 'vitesse'
                             else:
                                 profil = 'endurance'
                    #eval vma

                   # 1) Récupérer le dernier run (adapter le nom d'expérience si besoin)
                    runs = mlflow.search_runs(
                    experiment_names=["pipeline"], order_by=["start_time desc"], max_results=1
                    )
                    run_id = runs.iloc[0]["run_id"]

                    # 2) Télécharger l'artefact 'model_pipeline/model_pipeline.pkl'
                    client = MlflowClient()
                    local_dir = client.download_artifacts(run_id, "model_pipeline", dst_path="./restore")
                    pkl_path = os.path.join(local_dir, "model_pipeline.pkl")

                    # 3) Charger la pipeline
                    pipeline = joblib.load(pkl_path)   # ou: pickle.load(open(pkl_path, "rb"))
                    VMA = find_distance_bisection(pipeline, score_athlete, sexe_pred)
                    VMA = VMA * 10
                    
                    
                    row = {
                    "Name": athlete_prog,
                    "VMA": VMA,
                    "profil": profil,
                    "semaineprogramme": nombresemmaine,
                    "nbkilosemaine": nombrekilos,
                    "nbentrainementsemaine": nombreentrainement,
                    "niveauprogramme": niveau,
                    "distancecible": distance_pred                    
}
                    
                    exp_nameid = 'Programme'
                    pdf = get_data("data/pdf.pkl", exp_nameid)
                    athlete_pdf = pdf[pdf["Name"] == athlete_prog]
                   
                    
                    if len(athlete_pdf) > 0 :
                         dist_old = int(athlete_pdf['distancecible'].iloc[0])
                         dist_new = int(distance_pred)
                         st.success(f"Programme {dist_old} km remplacé par un programme sur {dist_new} km")
                         # Trouver l'index de la ligne à modifier
                         idx = pdf.index[pdf["Name"] == athlete][0]
                         # Modifier la ligne
                         pdf.loc[idx, :] = row
                    else:
                         dist_new = int(distance_pred)
                         st.success(F"Programme {dist_new} km créé")
                         pdf = pd.concat([pdf, pd.DataFrame([row])], ignore_index=True)
                    pdf.to_csv("data.csv", index=False)
                    exp_nameid = 'Programme'
                    nrun_name = 'data'+str(athlete_prog) + str(exp_nameid)
                    log_dataframe_to_mlflow(pdf, nrun_name, exp_nameid)
                    # création du programme
    pdf = get_data("data/pdf.pkl", exp_nameid)
    athlete_pdf = pdf[pdf["Name"] == athlete_prog]
    if athlete_pdf.empty:
                    st.markdown("Vous n'avez pas encore de programme d'entrainement") 
    else:
                #initvariable
                    
                    athlete = athlete_prog
                    VMA = athlete_pdf['VMA'].iloc[0]
                    profil = athlete_pdf['profil'].iloc[0]
                    nombresemmaine = athlete_pdf['semaineprogramme'].iloc[0]
                    nombrekilos = athlete_pdf['nbkilosemaine'].iloc[0]
                    nombreentrainement = athlete_pdf['nbentrainementsemaine'].iloc[0]
                    niveau = athlete_pdf['niveauprogramme'].iloc[0]
                    distance_pred = athlete_pdf['distancecible'].iloc[0]
                              


       
    # recherche nombrede seance par type
                    #sortilongue
                    longuesemaine = 1
                    #fractionné
                    if nombreentrainement == 3 or nombreentrainement == 4:
                          if distance_pred == 5 :
                               typefractionne = 'court'
                               fractionnesemaine = 1
                          else:
                               typefractionne = 'long'
                               fractionnesemaine = 1
                    if  nombreentrainement == 5:
                          if profil == 'endurance':
                              if distance_pred == 5 or distance_pred == 10:
                                  fractionnesemaine = 2
                              else:
                                  fractionnesemaine = 1
                                  typefractionne = 'long'
                          else:
                              fractionnesemaine = 1
                              typefractionne = 'long'

                    if nombreentrainement == 6:
                          if profil == 'vitesse':
                              if distance_pred == 42 and nombrekilos < 60:
                                  fractionnesemaine = 1
                                  typefractionne = 'long'
                              else:
                                  fractionnesemaine = 2
                          else:
                              fractionnesemaine = 2

                    
              
                    
                    


                    newnombrekilo = nombrekilos
                    sfractsemaine = fractionnesemaine
                    st.title(f"Programme pour préparer un {int(distance_pred)} km pour {athlete_prog}")
                    for semaine in range(1, nombresemmaine + 1):

                    #calcul kilometragedelasemaine
                      if semaine == 1:
                         newnombrekilo = newnombrekilo
                      else:
                        if niveau == 'Facile':
                          if semaine % 4 != 0 and semaine != nombresemmaine :
                              newnombrekilo = newnombrekilo + (5 * newnombrekilo/100)
                              fractionnesemaine = sfractsemaine
                          else:
                              newnombrekilo = newnombrekilo * 0.9
                              fractionnesemaine = 0
                        if niveau == 'Intermédiaire':
                          if semaine % 4 != 0 and semaine != nombresemmaine:
                              newnombrekilo = newnombrekilo + (8 * newnombrekilo/100)
                              fractionnesemaine = sfractsemaine
                          else:
                              newnombrekilo = newnombrekilo * 0.85
                              fractionnesemaine = 0
                        if niveau == 'Difficile':
                          if semaine % 4 != 0 and semaine != nombresemmaine :
                              newnombrekilo = newnombrekilo + (10 * newnombrekilo/100)
                              fractionnesemaine = sfractsemaine
                          else:
                              newnombrekilo = newnombrekilo * 0.8
                              fractionnesemaine = 0
                    #calcul kilometre séance
                    #fraction
                      tierprogramme = (semaine - 1) * 3 // nombresemmaine + 1
                      footingsemaine = nombreentrainement - longuesemaine - fractionnesemaine
                      nbechauffement = fractionnesemaine
                      nbretouraucalme = fractionnesemaine
                      echauffementkilo = 2
                      retouraucalmekilo = 2
                      
                      if niveau == 'Facile':
                        if fractionnesemaine == 0:
                            kilofractionnelong = 0
                            kilofractionnecourt = 0
                        if fractionnesemaine == 1:
                            if typefractionne =='court':
                               kilofractionnecourt = newnombrekilo * 15 / 100
                               if kilofractionnecourt > 1.5:
                                  kilofractionnecourt = 1.5
                               kilofractionnelong = 0
                            if typefractionne =='long':
                               kilofractionnelong = newnombrekilo * 15 / 100 
                               if kilofractionnelong > 6:
                                  kilofractionnelong = 6
                               kilofractionnecourt = 0
                        if fractionnesemaine == 2:
                             kilofract = newnombrekilo * 20 / 100
                             kilofractionnecourt = kilofract * 40 / 100
                             if kilofractionnecourt > 1.5:
                                 kilofractionnecourt = 1.5
                             kilofractionnelong = kilofract - kilofractionnecourt
                             if kilofractionnelong > 6:
                                  kilofractionnelong = 6
                      if niveau == 'Intermédiaire':
                        if fractionnesemaine == 0:
                            kilofractionnelong = 0
                            kilofractionnecourt = 0
                        if fractionnesemaine == 1:
                            if typefractionne =='court':
                               kilofractionnecourt = newnombrekilo * 20 / 100
                               if kilofractionnecourt > 2:
                                  kilofractionnecourt = 2 
                               kilofractionnelong = 0
                            if typefractionne =='long':
                               kilofractionnelong = newnombrekilo * 20 / 100 
                               if kilofractionnelong > 8:
                                   kilofractionnelong = 8
                               kilofractionnecourt = 0
                        if fractionnesemaine == 2:
                             kilofract = newnombrekilo * 25 / 100
                             kilofractionnecourt = kilofract * 40 / 100
                             if kilofractionnecourt > 2:
                                kilofractionnecourt = 2 
                             kilofractionnelong = kilofract - kilofractionnecourt
                             if kilofractionnelong > 8:
                                kilofractionnelong = 8
                      if niveau == 'Difficile':
                        if fractionnesemaine == 0:
                            kilofractionnelong = 0
                            kilofractionnecourt = 0
                        kilofractionne = newnombrekilo * 25 / 100
                        if fractionnesemaine == 1:
                            if typefractionne =='court':
                               kilofractionnecourt = newnombrekilo * 25 / 100
                               if kilofractionnecourt > 3:
                                  kilofractionnecourt = 3
                               kilofractionnelong = 0
                            if typefractionne =='long':
                               kilofractionnelong = newnombrekilo * 25 / 100 
                               if kilofractionnelong > 10 :
                                  kilofractionnelong = 10
                               kilofractionnecourt = 0
                        if fractionnesemaine == 2:
                             kilofract = newnombrekilo * 25 / 100
                             kilofractionnecourt = kilofract * 40 / 100
                             if kilofractionnecourt > 3 :
                                  kilofractionnecourt = 3
                             kilofractionnelong = kilofract - kilofractionnecourt
                             if kilofractionnelong > 10:
                                kilofractionnelong = 10 
                    #sortielongue
                       
                      
                      if distance_pred == 42:
                           kilosortielongue = newnombrekilo  * ((1 / nombreentrainement)  + 0.15)
                      if distance_pred == 21:
                           kilosortielongue = newnombrekilo  * ((1 / nombreentrainement)  + 0.10)
                      if distance_pred == 10:
                           kilosortielongue = newnombrekilo  * ((1 / nombreentrainement)  + 0.075)
                      kilofooting = (newnombrekilo - kilosortielongue - kilofractionnecourt - kilofractionnelong - (nbretouraucalme * retouraucalmekilo) - (nbechauffement * echauffementkilo)) / footingsemaine
                     
                    #calcul vitesse
                      if niveau == 'Facile':
                        Vsortielongue = VMA * 57 / 100
                        Vfooting = VMA * 57 / 100
                        Vfractionne100 = VMA * 92 / 100 
                        Vfractionne200 = VMA * 92 / 100 
                        Vfractionne300 = VMA * 89 / 100 
                        Vfractionne400 = VMA * 87 / 100 
                        Vfractionne500 = VMA * 85 / 100 
                        Vfractionne800 = VMA * 81 / 100 
                        Vfractionne1000 = VMA * 80 / 100 
                        Vfractionne2000 = VMA * 76 / 100 
                      if niveau == 'Intermédiaire':
                        Vsortielongue = VMA * 60 / 100
                        Vfooting = VMA * 60 / 100
                        Vfractionne100 = VMA * 95 / 100 
                        Vfractionne200 = VMA * 95 / 100 
                        Vfractionne300 = VMA * 92 / 100 
                        Vfractionne400 = VMA * 90 / 100 
                        Vfractionne500 = VMA * 88 / 100 
                        Vfractionne800 = VMA * 84 / 100 
                        Vfractionne1000 = VMA * 82 / 100 
                        Vfractionne2000 = VMA * 78 / 100 
                      if niveau == 'Difficile':
                        Vsortielongue = VMA * 65 / 100
                        Vfooting = VMA * 65 / 100
                        Vfractionne100 = VMA * 100 / 100 
                        Vfractionne200 = VMA * 100 / 100 
                        Vfractionne300 = VMA * 96 / 100 
                        Vfractionne400 = VMA * 94 / 100 
                        Vfractionne500 = VMA * 92 / 100 
                        Vfractionne800 = VMA * 87 / 100 
                        Vfractionne1000 = VMA * 85 / 100 
                        Vfractionne2000 = VMA * 81 / 100 
                     # preparation du programme
                     #semaine 1
                      st.title(f"Semaine {semaine}/{nombresemmaine}")
                      st.subheader("Lundi")
                      st.write("Repos")
                      st.subheader("Mardi")
                      if fractionnesemaine == 0:
                         minute, seconde = kmh_to_pace(Vfooting)
                         st.write(F"Footing :{round(kilofooting, 1)} KM à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                      else : 
                        if fractionnesemaine == 2 or (fractionnesemaine == 1 and typefractionne =='court'):
                             if tierprogramme == 1:
                                 if semaine % 2 == 1:
                                     nombrerepetition = round(kilofractionnecourt / 100 * 1000)
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     if nombrerepetition > 15:
                                         if nombrerepetition % 2 != 0:
                                            nombrerepetition = nombrerepetition + 1
                                         repetitionbloc = nombrerepetition / 2
                                         repetitionbloc = int(round(repetitionbloc))
                                         minute, seconde = kmh_to_pace(Vfractionne100)
                                         st.write(F"Fractionné court : 2 blocs de {repetitionbloc} × 100 m à {round(Vfractionne100, 2)} km/h({minute}m{str(seconde).zfill(2)} le km)")
                                     else :
                                         minute, seconde = kmh_to_pace(Vfractionne100)
                                         st.write(F"Fractionné court : {nombrerepetition} × 100 m à {round(Vfractionne100, 2)} km/h({minute},{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 else:
                                     nombrerepetition = round(kilofractionnecourt / 200 * 1000)
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfractionne200)
                                     st.write(F"Fractionné court : {nombrerepetition} × 200 m à {round(Vfractionne200, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                             if tierprogramme == 2:
                                 if semaine % 2 == 1:
                                     nombrerepetition = round(kilofractionnecourt / 300 * 1000)
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfractionne300)
                                     st.write(F"Fractionné court : {nombrerepetition} ×  300m à {round(Vfractionne300, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 else:
                                     nombrerepetition = round(kilofractionnecourt / 400 * 1000)
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfractionne400)
                                     st.write(F"Fractionné court : {nombrerepetition} × 400 m à {round(Vfractionne400, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                             if tierprogramme == 3:
                                 if semaine % 2 == 1:
                                     nombrerepetition = round(kilofractionnecourt / 400 * 1000)
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfractionne400)
                                     st.write(F"Fractionné court : {nombrerepetition} × 400 m à {round(Vfractionne400, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 else:
                                     nombrerepetition = round(kilofractionnecourt / 500 * 1000)
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km d'echauffement à {Vfooting} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfractionne500)
                                     st.write(F"Fractionné court : {nombrerepetition} × 500 m à {round(Vfractionne500, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                     minute, seconde = kmh_to_pace(Vfooting)
                                     st.write(F"2 km de retour au calme à {Vfooting} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                        else:
                             if tierprogramme == 1:
                                 nombrerepetition = round(kilofractionnelong / 800 * 1000)
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfractionne800)
                                 st.write(F"Fractionné long : {nombrerepetition} × 800 m à {round(Vfractionne800, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                             if tierprogramme == 2:
                                 nombrerepetition = round(kilofractionnelong / 1000 * 1000)
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfractionne1000)
                                 st.write(F"Fractionné long : {nombrerepetition} × 1000 m à {round(Vfractionne1000, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                             if tierprogramme == 3:
                                 nombrerepetition = round(kilofractionnelong / 2000 * 1000)
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfractionne2000)
                                 st.write(F"Fractionné long : {nombrerepetition} × 2000 m à {round(Vfractionne2000, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                      st.subheader("Mercredi")
                      if nombreentrainement > 4:
                         minute, seconde = kmh_to_pace(Vfooting)
                         st.write(F"Footing :{round(kilofooting, 1)} KM à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                      else:
                         st.write("Repos")
                      st.subheader("Jeudi")
                      minute, seconde = kmh_to_pace(Vfooting)
                      st.write(F"Footing :{round(kilofooting, 1)} KM à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                      st.subheader("Vendredi")
                      if nombreentrainement > 3:
                         if fractionnesemaine == 2:
                             if tierprogramme == 1:
                                 nombrerepetition = round(kilofractionnelong / 800 * 1000)
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfractionne800)
                                 st.write(F"Fractionné long : {nombrerepetition} × 800 m à {round(Vfractionne800, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                             if tierprogramme == 2:
                                 nombrerepetition = round(kilofractionnelong / 1000 * 1000)
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfractionne1000)
                                 st.write(F"Fractionné long : {nombrerepetition} × 1000 m à {round(Vfractionne1000, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                             if tierprogramme == 3:
                                 nombrerepetition = round(kilofractionnelong / 2000 * 1000)
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km d'echauffement à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfractionne2000)
                                 st.write(F"Fractionné long : {nombrerepetition} × 2000 m à {round(Vfractionne2000, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                                 minute, seconde = kmh_to_pace(Vfooting)
                                 st.write(F"2 km de retour au calme à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                         else:
                               minute, seconde = kmh_to_pace(Vfooting)
                               st.write(F"Footing :{round(kilofooting, 1)} KM à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                      else:
                         st.write("Repos")
                      st.subheader("Samedi")
                      if nombreentrainement > 5:
                        minute, seconde = kmh_to_pace(Vfooting)
                        st.write(F"Footing :{round(kilofooting, 1)} KM à {round(Vfooting, 2)} km/h({minute}m{str(seconde).zfill(2)}s le km)")
                      else:
                        st.write("Repos")
                      st.subheader("Dimanche")
                      minute, seconde = kmh_to_pace(Vsortielongue)
                      st.write(F"Sortie longue : {round(kilosortielongue, 1)} KM à {round(Vsortielongue, 2)} km/h({minute}m{str(str(seconde).zfill(2))}s le km)")
if st.session_state.page == "contact":
    athlete = st.session_state.athlete
    st.title("📞 Contact")
    st.markdown(
        """
        <p>Vous pouvez me contacter ou en savoir plus via les liens ci-dessous :</p>

        <ul>
            <li>🌐 <b>Portfolio :</b> <a href="https://portfoliobenoitgoffinet-bmcuhjbxfhbsc4ae.canadacentral-01.azurewebsites.net/" target="_blank">portfolio/Datascience</a></li>
            <li>✉️ <b>Email :</b> <a href="benoitgoffinet@live.fr">benoitgoffinet@live.fr</a></li>
            <li>💼 <b>LinkedIn :</b> <a href="https://www.linkedin.com/in/benoit-goffinet/" target="_blank">Linkedin/BenoitGoffinet</a></li>
            <li>💼 <b>Github :</b> <a href="https://github.com/benoitgoffinet" target="_blank">Github/Datascience</a></li>
        </ul>

        <hr>
        
        """,
        unsafe_allow_html=True
    )
    message = st.text_area(
    "Votre message :", 
    placeholder="Écrivez ici votre message...",
    height=200)
    if st.button("Envoyer le message"):
      if message.strip():
        st.success("Message envoyé ✅")
        exp_namemessage = "Message"
        mlflow.set_experiment(exp_namemessage)
        with mlflow.start_run(run_name=f"message_{athlete}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("utilisateur", athlete)
            mlflow.log_param("message", message)
            mlflow.log_param("date", str(datetime.now()))
        st.session_state.identification_idf = idf
        idf["messageenvoye"] = idf["messageenvoye"].fillna(0)
        idf.loc[idf["Name"] == athlete, "messageenvoye"] = (
        idf.loc[idf["Name"] == athlete, "messageenvoye"] + 1
             )
        nrun_name = f"data_{athlete}_{1}"
        log_dataframe_to_mlflow(idf, nrun_name, exp_nameid)
        st.session_state.identification_idf = idf
      else:
        st.warning("Veuillez écrire un message avant d’envoyer.")

    # --- 🔍 Affichage des réponses précédentes ---
    st.divider()
    st.subheader("📨 Vos messages précédents et réponses")

    mlflow.end_run()  # ferme tout run actif
    exp_name = "Message"
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)

    if exp is not None:
    # Récupère tous les messages de l'utilisateur actuel
      runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"params.utilisateur = '{athlete}'",
        order_by=["attributes.start_time DESC"]
    )

      if not runs:
        st.info("Vous n'avez pas encore envoyé de message.")
      else:
        for run in runs:
            params = run.data.params
            msg = params.get("message", "—")
            date_msg = params.get("date", "—")
            reponse = params.get("reponse", "⏳ En attente de réponse")
            date_rep = params.get("date_reponse", None)

            with st.expander(f"🗓️ Message du {date_msg}"):
                st.write(f"**Votre message :** {msg}")
                if reponse != "⏳ En attente de réponse":
                    st.success(f"**Réponse :** {reponse}")
                    if date_rep:
                        st.caption(f"📅 Répondu le {date_rep}")
                else:
                    st.info(reponse)
    else:
      st.warning("Aucune expérience 'Message' trouvée dans MLflow.")

        

