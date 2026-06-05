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
# CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Prédiction PV par IA", page_icon="☀️", layout="wide")

# Initialisation de la session
if "donnees" not in st.session_state: st.session_state.donnees = None
if "page" not in st.session_state: st.session_state.page = "🏠 Accueil & Présentation"

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🗂️ Menu Principal")
    # On met à jour la session à chaque sélection
    st.session_state.page = st.radio(
        "Sélectionnez une page :", 
        ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"]
    )
    st.divider()
    # Configuration IA
    nom_modele_selectionne = st.selectbox("Modèle d'IA actif :", ["LSTM", "GRU", "MLP", "ARX", "ANFIS"])

# =============================================================================
# LOGIQUE DES PAGES
# =============================================================================

# PAGE 1 : ACCUEIL
if st.session_state.page == "🏠 Accueil & Présentation":
    if os.path.exists("panneau_pv.jpg"):
        st.image("panneau_pv.jpg", use_container_width=True)
    st.markdown("<div style='text-align: center;'><h1>Prédiction de la Production Photovoltaïque</h1></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 👥 Réalisé par : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI")
    with col2:
        st.markdown("### 👨‍🏫 Encadré par : M. Anisse CHIALI & Mme Imane NEDJAR")

# PAGE 2 : IMPORTATION
elif st.session_state.page == "📂 Importation des Données":
    st.title("📂 Importation des Données")
    fichier = st.file_uploader("Choisir fichier :", type=["xlsx", "csv"])
    if fichier:
        df = pd.read_csv(fichier) if fichier.name.endswith(".csv") else pd.read_excel(fichier)
        df = df.drop(columns=["Unnamed: 8"], errors="ignore")
        st.session_state.donnees = df
        st.dataframe(df.head())
        st.subheader("📊 Statistiques (Min/Max/Moyenne)")
        st.write(df.describe().loc[['min', 'max', 'mean']])

# PAGE 3 : ÉVALUATION
elif st.session_state.page == "📊 Évaluation & Graphiques":
    st.title("📊 Évaluation")
    if st.session_state.donnees is not None:
        df = st.session_state.donnees.dropna()
        # Simulation split
        n = len(df)
        # Affichage graphe (Remplacer y_reel/y_pred par vos vrais modèles)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df.iloc[:, -1], name="Réel"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Veuillez charger des données.")

# PAGE 4 : PRÉDICTION
elif st.session_state.page == "🔮 Prédiction Future":
    st.title("🔮 Prédiction Future")
    v1 = st.slider("LDR_Raw", 0, 4095, 1500)
    v2 = st.slider("Humidité", 0.0, 100.0, 50.0)
    v3 = st.slider("Température", -5.0, 50.0, 25.0)
    
    # Courbe unique de prédiction
    fig_futur = go.Figure()
    fig_futur.add_trace(go.Scatter(y=[0.5, 0.8, 0.7], name="Prédiction pure", line=dict(color="#FF6B2B")))
    st.plotly_chart(fig_futur, use_container_width=True)
