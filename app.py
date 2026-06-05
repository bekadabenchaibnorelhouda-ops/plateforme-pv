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

warnings.filterwarnings("ignore")

# --- CONFIGURATION (Votre base) ---
st.set_page_config(page_title="Prédiction PV par IA", layout="wide")
COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C"]
COLONNE_CIBLE = "Puissance_mW"

# --- CHARGEMENT ---
@st.cache_resource
def charger_ressources():
    # Ici, remettez votre logique de chargement de modèles
    return {}, None

modeles, scaler = charger_ressources()

# --- INITIALISATION ---
if "donnees" not in st.session_state: st.session_state.donnees = None
if "page" not in st.session_state: st.session_state.page = "🏠 Accueil & Présentation"

# --- SIDEBAR (Votre structure) ---
with st.sidebar:
    page = st.radio("Menu Principal :", ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"])
    nom_modele_sel = st.selectbox("Modèle IA :", ["LSTM", "GRU", "MLP", "ARX", "ANFIS"])
    cle_modele = nom_modele_sel.lower()

# --- PAGE 2 : IMPORTATION (Restauration stricte) ---
if page == "📂 Importation des Données":
    st.title("📂 Importation des Données Capteurs")
    fichier = st.file_uploader("Choisir votre fichier :", type=["xlsx", "csv"])
    if fichier:
        df = pd.read_csv(fichier) if fichier.name.endswith(".csv") else pd.read_excel(fichier)
        df = df.drop(columns=["Unnamed: 8"], errors="ignore") # Suppression colonne inutile
        st.session_state.donnees = df
        st.dataframe(df.head(10))
        # AJOUT DES STATS DEMANDÉES
        st.subheader("📊 Statistiques Min/Max")
        st.write(df.describe().loc[['min', 'max', 'mean']])

# --- PAGE 3 : ÉVALUATION (Calcul dynamique) ---
elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Évaluation")
    if st.session_state.donnees is not None:
        df = st.session_state.donnees.dropna()
        X = df[COLONNES_REQUISES].values
        y_reel = df[COLONNE_CIBLE].values
        
        # Inférence dynamique
        model = modeles.get(cle_modele)
        if model:
            y_pred = model.predict(X).flatten()
            # Métriques
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_reel, y_pred)):.4f}")
            c2.metric("MAE", f"{mean_absolute_error(y_reel, y_pred):.4f}")
            c3.metric("R²", f"{r2_score(y_reel, y_pred):.4f}")
            
            # Graphique
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_reel, name="Réel"))
            fig.add_trace(go.Scatter(y=y_pred, name="Prédit"))
            st.plotly_chart(fig, use_container_width=True)

# --- PAGE 4 : PRÉDICTION (Calcul dynamique) ---
elif page == "🔮 Prédiction Future":
    st.title("🔮 Prédiction Future")
    l = st.slider("LDR", 0, 4095, 1500)
    h = st.slider("Hum", 0.0, 100.0, 50.0)
    t = st.slider("Temp", -5.0, 50.0, 25.0)
    
    if modeles.get(cle_modele):
        pred = modeles[cle_modele].predict(np.array([[l, h, t]]))[0]
        st.metric("Puissance", f"{pred:.2f} mW")
        
        # Graphique Test (15%)
        st.subheader("Graphe de Test (15% derniers points)")
        data_test = st.session_state.donnees.iloc[-int(len(st.session_state.donnees)*0.15):]
        preds = modeles[cle_modele].predict(data_test[COLONNES_REQUISES].values)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=preds.flatten(), mode='lines+markers', name="Prédictions"))
        st.plotly_chart(fig, use_container_width=True)
