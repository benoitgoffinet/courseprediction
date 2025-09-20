import streamlit as st
import pandas as pd
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

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if "submit_data" not in st.session_state:
    st.session_state.submit_data = False


st.title("Application de suivi et prédiction de performances de course ")


# Initialisation ou chargement du DataFrame
FILE_PATH = "ndf.pkl"

# --- Chargement ou initialisation ---
if "ndf" not in st.session_state:
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "rb") as f:
            st.session_state.ndf = pickle.load(f)
    else:
        st.session_state.ndf = pd.DataFrame(columns=[
            "athlete", 
            "distance (m)", 
            "elevation gain (m)", 
            "gender", 
            "elapsed time (s)"
        ])

# --- Formulaire pour ajouter une nouvelle performance ---
st.subheader("Ajouter une performance – Plus vous enregistrez de courses, plus les prédictions seront précises (minimum 2)")
with st.form("add_performance"):
  athlete = st.text_input("Nom de l'athlète")
  distance = st.number_input("Distance (km)", min_value=0.0, value=5.0)
  denivele = st.number_input("Dénivelé (m)", min_value=0, value=0)
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
     'distance (m)':[distance * 1000],
     'elevation gain (m)':[denivele],
     'gender':[sexe],
     'elapsed time (s)':[minutes * 60 + secondes]
    })
    st.session_state.ndf = pd.concat([st.session_state.ndf, new_row], ignore_index=True)
    st.success(f"Performance ajoutée pour {athlete}")
    st.session_state.submit_data = True


# Affichage du DataFrame
if st.session_state.submit_data == True:
   athlete_df = st.session_state.ndf[st.session_state.ndf["athlete"] == athlete]

# Sélectionner seulement les colonnes concernées
   cols_to_display = ["athlete", "distance (m)", "elevation gain (m)", "gender", "elapsed time (s)"]
   athlete_df = athlete_df[cols_to_display]
   st.subheader(f"Historique des performances de {athlete}")
   st.dataframe(athlete_df)


# --- Réentraînement du modèle si données suffisantes ---
# Filtrer le DataFrame pour l'athlète sélectionné
athlete_df = st.session_state.ndf[st.session_state.ndf["athlete"] == athlete]

# Vérifier qu'il a au moins 2 performances
if len(athlete_df) < 2:
    st.warning(f"{athlete} n'a pas encore assez de performances pour la prédiction.")
else:
    st.write(f"{athlete} a au moins 2 performances, on peut entraîner le modèle.")
     # Bouton pour entraîner le modèle
    if st.button("Entraîner le modèle"):
       st.session_state.model_trained = True
        # Variables explicatives et cible
       #choix des variables explicatives
       st.session_state.ndf['athlete'] = st.session_state.ndf['athlete'].astype(str)
       dataexplicative = st.session_state.ndf.drop(columns=["T/K", 'timestamp', 'elapsed time (s)', 'M/S', 'K/H', 'average heart rate (bpm)'])
       target = st.session_state.ndf['elapsed time (s)'] 
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
       'model__min_samples_split': [1, 2]
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

# 🚀 Entraînement avec GridSearchCV
       grid_search.fit(X_train, y_train)

# ✅ Meilleur modèle trouvé
       best_model = grid_search.best_estimator_
       st.session_state.best_model = best_model

       print(f"✅ Meilleur alpha trouvé : {grid_search.best_params_}")
       print(f"📊 Meilleur score R² (validation croisée) : {grid_search.best_score_:.4f}")

# 🟢 Prédictions sur l'ensemble de test
       y_pred = best_model.predict(X_test)

# 📊 Évaluation
       r2_test = r2_score(y_test, y_pred)
       mse_test = mean_squared_error(y_test, y_pred)

       print(f"📊 R² Score (test) : {r2_test:.4f}")
       print(f"📉 Mean Squared Error (test) : {mse_test:.4f}")

# 🟢 Prédictions sur l'ensemble d'entraînement
       y_pred_train = best_model.predict(X_train)

# 📊 Évaluation sur l'ensemble d'entraînement
       r2_train = r2_score(y_train, y_pred_train)
       mse_train = mean_squared_error(y_train, y_pred_train)

       print(f"📊 R² Score (train) : {r2_train:.4f}")
       print(f"📉 Mean Squared Error (train) : {mse_train:.4f}")
        
# Formulaire prédiction
if st.session_state.model_trained == True:
       with st.form("predict_time"):
         athlete_pred = st.selectbox("Sélectionnez l'athlète", st.session_state.ndf['athlete'].unique())
         distance_pred = st.number_input("Distance (km)", min_value=0.0, value=5.0, key='dist_pred')
         denivele_pred = st.number_input("Dénivelé (m)", min_value=0.0, value=0.0, key='den_pred')
         sexe_pred = st.selectbox("Sexe", options=['M','F'], key='sexe_pred')
         predict_submitted = st.form_submit_button("Prédire le temps")


       if predict_submitted:
             X_new = pd.DataFrame({
                  "athlete": [athlete_pred],
                  "distance (m)": [distance_pred * 1000],          # km -> m
                  "elevation gain (m)": [denivele_pred],
                  "gender": [sexe_pred]
         })

             X_new['athlete'] = X_new['athlete'].astype(str)
             X_new['gender'] = X_new['gender'].astype(str)
             predicted_time = st.session_state.best_model.predict(X_new)[0]
             minutes = int(predicted_time // 60)
             seconds = int(predicted_time % 60)
             st.success(f"Temps prédit pour {athlete_pred} : {minutes} min {seconds} sec")