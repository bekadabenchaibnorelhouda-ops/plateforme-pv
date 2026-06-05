import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION INITIALE ---
st.set_page_config(page_title="Plateforme PV", layout="wide")

# Simulation de chargement de modèles (Remplacez par vos vrais chemins)
# Assurez-vous que ces fichiers .pkl/.h5 sont bien dans votre dossier GitHub
dict_modeles = {}
if os.path.exists("model_arx.pkl"): dict_modeles["ARX"] = joblib.load("model_arx.pkl")
# Ajoutez ici le chargement de vos autres modèles (MLP, ANFIS, LSTM, etc.)

# --- 2. NAVIGATION ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Aller vers :", ["Accueil", "Données", "Évaluation", "Prédiction"])

# Initialisation session_state
if "df_user" not in st.session_state: st.session_state["df_user"] = None

# --- 3. LOGIQUE DES SECTIONS ---

if section == "Accueil":
    st.title("Bienvenue sur la plateforme PV")

elif section == "Données":
    st.title("Importation des données")
    file = st.file_uploader("Chargez votre fichier CSV", type=["csv"])
    if file: st.session_state["df_user"] = pd.read_csv(file)

elif section == "Évaluation":
    st.title("Évaluation des modèles")
    st.write("Section des graphiques et scores ici.")

elif section == "Prédiction":
    st.title("Prévision de la Puissance")
    
    # Sliders de contrôle
    ldr = st.slider("Éclairement (LDR_Raw)", 0, 2000, 500)
    hum = st.slider("Humidité (Hum_%)", 0, 100, 50)
    tmp = st.slider("Température (Temp_C)", -10, 50, 25)
    
    modele_choisi = st.selectbox("Choisir un modèle", list(dict_modeles.keys()))
    
    if st.button("Prédire"):
        input_data = np.array([[ldr, hum, tmp]])
        model = dict_modeles[modele_choisi]
        
        try:
            # Gestion dimension pour LSTM/GRU
            if "LSTM" in modele_choisi or "GRU" in modele_choisi:
                input_data = input_data.reshape((1, 1, 3))
                
            pred = model.predict(input_data)
            valeur = float(np.array(pred).flatten()[0])
            
            st.metric("Puissance Prévue", f"{max(0.0, valeur):.2f} mW")
        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")

# --- 4. FIN DU SCRIPT ---
