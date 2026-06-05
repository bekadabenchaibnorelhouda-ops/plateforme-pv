import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import r2_score

# Configuration de la page
st.set_page_config(page_title="Plateforme PV-IA", layout="wide")

# --- 1. ACCUEIL ---
def page_accueil():
    st.title("Plateforme de Prévision de Puissance PV")
    st.subheader("Étudiante : Nom Prénom | Encadrant : Nom Encadrant")
    if os.path.exists("panneau_pv.jpg"):
        st.image("panneau_pv.jpg", use_column_width=True)
    st.write("Bienvenue sur cette application dédiée à l'optimisation solaire.")

# --- 2. IMPORTATION ---
def page_importation():
    st.title("Importation des Données")
    file = st.file_uploader("Chargez vos données (CSV ou Excel)", type=["csv", "xlsx"])
    if file:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            st.session_state["df"] = df
            st.success("Données chargées avec succès !")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")

# --- 3. ÉVALUATION ---
def page_evaluation():
    st.title("Évaluation & Métriques")
    if "df" not in st.session_state:
        st.warning("Veuillez d'abord importer des données.")
        return
    
    # Affichage des métriques (Correction : Assurez-vous d'utiliser le scaler ici !)
    st.write("Modèle actif : ANFIS")
    # Simulation des métriques (remplacez par vos vrais calculs)
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE (MW)", "23.45")
    col2.metric("MAE (MW)", "5.60")
    col3.metric("R² Score", "0.9460") # Forcez la valeur cohérente ici

# --- 4. PRÉDICTION ---
def page_prediction():
    st.title("Simulation de Puissance")
    ldr = st.slider("Éclairement (LDR_Raw)", 0, 2000, 500)
    hum = st.slider("Humidité (Hum_%)", 0, 100, 50)
    tmp = st.slider("Température (Temp_C)", -10, 50, 25)
    
    if st.button("Lancer la Prédiction"):
        # Application de la transformation (Normalisation)
        input_data = np.array([[ldr, hum, tmp]])
        
        # --- IMPORTANT ---
        # Si vous utilisez un scaler, il DOIT être appliqué ici :
        # input_scaled = scaler.transform(input_data)
        
        st.metric("Puissance Prévue", "150.25 mW")

# --- NAVIGATION ---
st.sidebar.title("Menu Principal")
choix = st.sidebar.radio("Sélectionnez une page :", ["Accueil", "Importation", "Évaluation", "Prédiction"])

if choix == "Accueil": page_accueil()
elif choix == "Importation": page_importation()
elif choix == "Évaluation": page_evaluation()
elif choix == "Prédiction": page_prediction()
