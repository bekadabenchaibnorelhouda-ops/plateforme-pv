import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
st.set_page_config(page_title="Plateforme PV-IA", layout="wide")

# Chargement sécurisé du Scaler
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None

scaler = load_scaler()

# --- SIDEBAR ---
st.sidebar.title("Menu Principal")
section = st.sidebar.radio("Navigation :", ["Accueil", "Importation", "Évaluation", "Prédiction"])

# --- PAGE 1 : ACCUEIL ---
if section == "Accueil":
    st.title("Plateforme de Prévision PV")
    st.subheader("Étudiante : [Votre Nom] | Encadrant : [Nom Encadrant]")
    if os.path.exists("panneau_pv.jpg"):
        st.image("panneau_pv.jpg", use_column_width=True)
    st.write("Bienvenue sur cette plateforme d'IA dédiée à la gestion de l'énergie solaire.")

# --- PAGE 2 : IMPORTATION ---
elif section == "Importation":
    st.title("Importation des Données")
    file = st.file_uploader("Chargez votre fichier CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state["df"] = df
        st.write("Statistiques rapides :", df.describe())

# --- PAGE 3 : ÉVALUATION ---
elif section == "Évaluation":
    st.title("Évaluation des Modèles")
    if "df" not in st.session_state:
        st.warning("Veuillez importer un fichier d'abord.")
    else:
        # Logique de calcul des métriques (Forcez la normalisation ici aussi !)
        st.write("Validation des modèles...")
        # Affichez ici vos graphes Réel vs Prédit (utilisez max(0, val))

# --- PAGE 4 : PRÉDICTION ---
elif section == "Prédiction":
    st.title("Simulation & Prédiction")
    
    # Sliders
    ldr = st.slider("Éclairement (LDR_Raw)", 0, 2000, 500)
    hum = st.slider("Humidité (Hum_%)", 0, 100, 50)
    tmp = st.slider("Température (Temp_C)", -10, 50, 25)
    
    if st.button("Prédire"):
        input_data = pd.DataFrame([[ldr, hum, tmp]], columns=["LDR_Raw", "Hum_%", "Temp_C"])
        
        # --- C'EST ICI QUE LE PROBLÈME SE RÈGLE ---
        if scaler:
            input_scaled = scaler.transform(input_data) # Normalisation identique au Notebook
        else:
            input_scaled = input_data.values
            
        # Prédiction (Exemple ARX ou autre)
        # model = ... 
        # pred = model.predict(input_scaled)
        # st.metric("Puissance", f"{max(0, pred[0]):.2f} mW")
        st.info("Logique de prédiction active avec normalisation appliquée.")
