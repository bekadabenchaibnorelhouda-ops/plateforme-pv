import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# 1. Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédiction PV par IA",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS original
st.markdown(
    """
    <style>
    :root {
        --couleur-primaire   : #FF6B2B;
        --couleur-secondaire : #F8F9FA;
        --couleur-accent     : #E65100;
        --couleur-fond       : #FFFFFF;
        --couleur-carte      : #F1F3F5;
        --couleur-texte      : #212529;
    }
    .stApp { background-color: var(--couleur-fond); color: var(--couleur-texte); font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: var(--couleur-secondaire); border-right: 1px solid #DEE2E6; }
    h1, h2, h3 { color: var(--couleur-accent) !important; font-weight: 700 !important; }
    
    .carte-metrique {
        background-color: var(--couleur-carte); border: 1px solid #CED4DA; border-radius: 12px;
        padding: 20px 24px; text-align: center; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); margin-bottom: 12px;
    }
    .carte-metrique .valeur { font-size: 2rem; font-weight: 700; color: var(--couleur-primaire); }
    .carte-metrique .label { font-size: 0.85rem; color: #495057; margin-top: 4px; text-transform: uppercase; }
    
    .cadre-accueil {
        background-color: #F8F9FA; border: 1px solid #DEE2E6; border-radius: 16px; padding: 25px; margin-bottom: 20px;
    }
    .badge-pfe {
        background-color: #E65100; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; display: inline-block; margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C", "Col_4", "Col_5"] # Liste étendue pour correspondre aux 5 colonnes attendues
COLONNE_CIBLE     = "Puissance_mW"

# ... (Le reste des fonctions charger_ressources et navigation reste identique)

# --- PAGE 2 : IMPORTATION ---
elif page == "📂 Importation des Données":
    st.title("📂 Importation des Données Capteurs")
    fichier_charge = st.file_uploader("Choisir votre fichier :", type=["xlsx", "csv"])
    if fichier_charge is not None:
        df = pd.read_csv(fichier_charge) if fichier_charge.name.endswith(".csv") else pd.read_excel(fichier_charge)
        # Création automatique des colonnes manquantes pour éviter l'erreur
        for col in COLONNES_REQUISES:
            if col not in df.columns:
                df[col] = 0.0
        st.session_state.donnees = df
        st.dataframe(df.head())
        st.subheader("📊 Statistiques")
        st.table(df.describe().loc[['min', 'max', 'mean', 'std']])

# --- PAGE 3 : ÉVALUATION (La partie corrigée) ---
elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Traitement & Évaluation")
    if st.session_state.donnees is not None:
        df = st.session_state.donnees.copy()
        # Assurer que les 5 colonnes existent
        for col in COLONNES_REQUISES:
            if col not in df.columns: df[col] = 0.0
            
        X = df[COLONNES_REQUISES].values
        y_reel = df[COLONNE_CIBLE].values
        
        try:
            # Transformation avec le scaler qui attend 5 colonnes
            X_scaled = scaler.transform(X) if scaler else X
            obj_modele = modeles.get(cle_modele)
            
            if cle_modele in ["gru", "lstm"]:
                y_pred = obj_modele.predict(X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]), verbose=0).flatten()
            else:
                y_pred = obj_modele.predict(X_scaled).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
            st.metric("RMSE", f"{rmse:.2f}")
        except Exception as e:
            st.error(f"Erreur technique : {e}")
