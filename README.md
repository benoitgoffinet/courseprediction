Application Streamlit – Prédiction de performances de course

Cette application permet d’enregistrer les performances de différents athlètes (distance, dénivelé, sexe, temps) et d’entraîner un modèle de machine learning pour prédire leurs futurs chronos.
L’utilisateur peut ajouter ses propres courses et obtenir des prédictions personnalisées.

🚀 Fonctionnalités

Ajouter de nouvelles performances (distance, sexe, temps).

Visualiser l’historique de ses performances.

Réentraîner le modèle automatiquement avec ses nouvelles données.

Prédire le temps sur une nouvelle distance/dénivelé.

traking et sauvegarde des datas avec mlflow

## 📦 Installation

Clone le dépôt et installe les dépendances :  
```bash
git clone https://github.com/benoitgoffinet/courseprediction.git
cd courseprediction
pip install -r requirements.txt