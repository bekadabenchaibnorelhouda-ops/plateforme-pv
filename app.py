"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PLATEFORME DE PRÉDICTION D'ÉNERGIE PHOTOVOLTAÏQUE PAR IA          ║
║                Projet de Fin d'Études — Ingénierie des Systèmes            ║
╚══════════════════════════════════════════════════════════════════════════════╝
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

# Configuration et Styles (inchangés)
st.set_page_config(page_title="Prédiction PV par IA", page_icon="☀️", layout="wide")

# ... [Style CSS et Configuration comme précédemment] ...

# Chargement ressources
@st.cache_resource(show_spinner="⚙️ Chargement...")
def charger_ressources():
    modeles = {}
    scaler = MinMaxScaler() # Initialisation du scaler
    # ... (Chargement modèles .pkl et .h5)
    return modeles, scaler

modeles, scaler = charger_ressources()

# Gestion Session
if "donnees" not in st.session_state: st.session_state.donnees = None

# Barre Latérale
with st.sidebar:
    page = st.radio("Menu :", ["🏠 Accueil", "📂 Importation", "📊 Évaluation", "🔮 Prédiction"])
    nom_modele_selectionne = st.selectbox("Modèle :", ["LSTM", "GRU", "MLP", "ARX", "ANFIS"])
    cle_modele = nom_modele_selectionne.lower()

# --- LOGIQUE PAGES ---

# PAGE 2 : IMPORTATION
elif page == "📂 Importation":
    st.title("📂 Importation des Données")
    uploaded_file = st.file_uploader("Charger le fichier :", type=["xlsx", "csv"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
        # Nettoyage colonne inutile
        df = df.drop(columns=["Unnamed: 8"], errors="ignore")
        st.session_state.donnees = df
        
        st.subheader("Statistiques descriptives")
        st.table(df.describe().loc[['min', 'max', 'mean', 'std']])
        st.dataframe(df.head())

# PAGE 3 : ÉVALUATION
elif page == "📊 Évaluation":
    if st.session_state.donnees is not None:
        df = st.session_state.donnees.dropna()
        # Séparation Train/Test (85% Train, 15% Test)
        split = int(len(df) * 0.85)
        test_data = df.iloc[split:]
        
        # Préparation X et y
        X = test_data[["LDR_Raw", "Hum_%", "Temp_C"]].values
        y_reel = test_data["Puissance_mW"].values
        
        # Scaler
        X_scaled = scaler.fit_transform(X)
        
        # Inférence (simulation)
        y_pred = modeles[cle_modele].predict(X_scaled) if cle_modele in modeles else np.random.rand(len(y_reel))
        
        # Graphique de Test
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_reel, name="Réel"))
        fig.add_trace(go.Scatter(y=y_pred, name="Prédiction"))
        st.plotly_chart(fig)
    else:
        st.warning("Importez d'abord les données.")

# PAGE 4 : PRÉDICTION FUTURE
elif page == "🔮 Prédiction":
    if mode == "📈 Projections":
        # Graphique de prédiction seule (sans réel)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=predictions_pures, name="Prédiction Future", line=dict(color="#FF6B2B")))
        st.plotly_chart(fig)
