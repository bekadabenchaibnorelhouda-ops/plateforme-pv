"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PLATEFORME DE PRÉDICTION D'ÉNERGIE PHOTOVOLTAÏQUE PAR IA          ║
║                Projet de Fin d'Études — Ingénierie des Systèmes            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Auteurs     : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI
Encadrement : M. Anisse CHIALI & Mme Imane NEDJAR
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(page_title="Prédiction PV par IA", page_icon="☀️", layout="wide")

# =============================================================================
# 2. STYLE CSS
# =============================================================================
st.markdown("""
    <style>
    .carte-metrique { background-color: #F1F3F5; border: 1px solid #CED4DA; border-radius: 12px; padding: 20px; text-align: center; }
    .valeur { font-size: 2rem; font-weight: 700; color: #FF6B2B; }
    .label { font-size: 0.85rem; color: #495057; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. CONFIGURATION & CHARGEMENT
# =============================================================================
COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C"]
COLONNE_CIBLE     = "Puissance_mW"
MODELES_DISPONIBLES = {
    "💾 LSTM": "lstm", "🔁 GRU": "gru", "🧠 MLP": "mlp", "📐 ARX": "arx", "🔮 ANFIS": "anfis"
}

@st.cache_resource(show_spinner="⚙️ Chargement...")
def charger_ressources():
    modeles = {}
    scaler = MinMaxScaler() # Initialisation standard
    # Note: Dans votre usage réel, chargez votre scaler entraîné ici si disponible
    return modeles, scaler

modeles, scaler = charger_ressources()

# =============================================================================
# 5. NAVIGATION
# =============================================================================
with st.sidebar:
    page = st.radio("Menu Principal :", ["🏠 Accueil", "📂 Importation", "📊 Évaluation", "🔮 Prédiction"])
    nom_modele = st.selectbox("Modèle actif :", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele]

# =============================================================================
# 6. PAGES
# =============================================================================

# PAGE 1 : ACCUEIL (Inchangée)
if page == "🏠 Accueil":
    st.title("Prédiction de la Production Photovoltaïque")
    st.write("Bienvenue sur la plateforme de prédiction.")

# PAGE 2 : IMPORTATION
elif page == "📂 Importation":
    st.title("📂 Importation des Données")
    fichier = st.file_uploader("Charger fichier :", type=["xlsx", "csv"])
    if fichier:
        df = pd.read_csv(fichier) if fichier.name.endswith(".csv") else pd.read_excel(fichier)
        df = df.drop(columns=["Unnamed: 8"], errors="ignore")
        st.session_state.donnees = df
        st.dataframe(df.head())
        st.subheader("📊 Statistiques Min/Max")
        st.write(df.describe().loc[['min', 'max', 'mean']])

# PAGE 3 : ÉVALUATION
elif page == "📊 Évaluation":
    st.title("📊 Évaluation des Modèles")
    if st.session_state.get("donnees") is not None:
        df = st.session_state.donnees.dropna()
        # Simulation d'évaluation
        y_reel = df[COLONNE_CIBLE].values
        y_pred = y_reel * 0.95 # Remplacer par model.predict(X)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_reel, name="Réel"))
        fig.add_trace(go.Scatter(y=y_pred, name="Prédiction"))
        st.plotly_chart(fig)
        st.success(f"R2 Score: {r2_score(y_reel, y_pred):.4f}")

# PAGE 4 : PRÉDICTION
elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction Future")
    v1 = st.slider("LDR_Raw", 0, 4095, 1000)
    v2 = st.slider("Humidité", 0, 100, 50)
    v3 = st.slider("Température", -5, 50, 25)
    
    # Prédiction pure (sans courbe réelle)
    pred = 100.0 # Résultat simulé de votre modèle
    st.metric("Puissance Estimée", f"{pred:.2f} mW")
