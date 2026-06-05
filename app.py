"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PLATEFORME DE PRÉDICTION D'ÉNERGIE PHOTOVOLTAÏQUE PAR IA            ║
║                Projet de Fin d'Études — Ingénierie des Systèmes              ║
╚══════════════════════════════════════════════════════════════════════════════╝
Auteurs  : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI
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

warnings.filterwarnings("ignore")

# 1. Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédiction PV par IA",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS conservé à l'identique
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

COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C"]
COLONNE_CIBLE      = "Puissance_mW"
FICHIER_EXEMPLE    = "Classeur1.xlsx"

MODELES_DISPONIBLES = {
    "💾 LSTM (Long Short-Term Memory)"          : "lstm",
    "🔁 GRU (Gated Recurrent Unit)"             : "gru",
    "🧠 MLP (Multi-Layer Perceptron)"           : "mlp",
    "📐 ARX (Auto-Regressive Exogenous)"        : "arx",
    "🔮 ANFIS (Adaptive Neuro-Fuzzy)"           : "anfis",
}

@st.cache_resource(show_spinner="⚙️ Chargement des architectures d'IA…")
def charger_ressources():
    modeles = {}
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    
    # Chargement ML
    for cle, nom in [("mlp", "model_mlp.pkl"), ("arx", "model_arx.pkl"), ("anfis", "model_anfis.pkl")]:
        if os.path.exists(nom): modeles[cle] = joblib.load(nom)
        
    # Chargement Keras
    try:
        from tensorflow.keras.models import load_model
        for cle, nom in [("gru", "model_gru.h5"), ("lstm", "model_lstm.h5")]:
            if os.path.exists(nom): modeles[cle] = load_model(nom, compile=False)
    except: pass
    return modeles, scaler

modeles, scaler = charger_ressources()

# Fonction utilitaire de préparation des données pour corriger les erreurs de dimension
def preparer_donnees(df, cle_modele):
    X = df[COLONNES_REQUISES].values.astype(float)
    if scaler is not None:
        X = scaler.transform(X)
    if cle_modele in ["gru", "lstm"]:
        X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X

# Initialisation session
if "donnees" not in st.session_state: st.session_state.donnees = None

with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 10px 0;'><div style='font-size:3rem;'>☀️</div></div>", unsafe_allow_html=True)
    page = st.radio("Sélectionnez une page :", ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"])
    st.divider()
    nom_modele_selectionne = st.selectbox("Modèle d'IA actif :", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele_selectionne]
    nom_court = nom_modele_selectionne.split("(")[0].strip()

# --- PAGES ---
if page == "🏠 Accueil & Présentation":
    st.markdown("""<div style="text-align: center;"><span class="badge-pfe">PROJET DE FIN D'ÉTUDES (PFE)</span><h1>Prédiction de la Production d'Énergie Photovoltaïque par IA</h1></div>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: st.markdown('<div class="cadre-accueil"><h4>👥 Auteurs :</h4> Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI<hr><h4>👨‍🏫 Encadrants :</h4> M. Anisse CHIALI & Mme Imane NEDJAR</div>', unsafe_allow_html=True)

elif page == "📂 Importation des Données":
    st.title("📂 Importation")
    fichier = st.file_uploader("Charger fichier :", type=["xlsx", "csv"])
    if fichier:
        st.session_state.donnees = pd.read_csv(fichier) if fichier.name.endswith('.csv') else pd.read_excel(fichier)
        st.success("Chargé !")

elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Évaluation des Métriques")
    if st.session_state.donnees is not None:
        df = st.session_state.donnees.dropna()
        X = preparer_donnees(df, cle_modele)
        y_reel = df[COLONNE_CIBLE].values
        
        y_pred = modeles[cle_modele].predict(X, verbose=0).flatten()
        
        # Correction des métriques : calcul rigoureux
        rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
        mae = mean_absolute_error(y_reel, y_pred)
        r2 = r2_score(y_reel, y_pred)
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="carte-metrique"><div class="valeur">{rmse:.2f}</div><div class="label">RMSE</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="carte-metrique"><div class="valeur">{mae:.2f}</div><div class="label">MAE</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="carte-metrique"><div class="valeur">{r2:.4f}</div><div class="label">R²</div></div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_reel[:200], name="Réel"))
        fig.add_trace(go.Scatter(y=y_pred[:200], name="IA"))
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Prédiction Future":
    st.title("🔮 Prédiction")
    val_ldr = st.slider("Éclairement", 0, 4095, 1500)
    val_hum = st.slider("Humidité", 0.0, 100.0, 65.0)
    val_temp = st.slider("Température", -5.0, 50.0, 25.0)
    
    # Correction : création d'un DataFrame de forme correcte
    df_input = pd.DataFrame([[val_ldr, val_hum, val_temp]], columns=COLONNES_REQUISES)
    X_input = preparer_donnees(df_input, cle_modele)
    
    pred = modeles[cle_modele].predict(X_input, verbose=0).flatten()[0]
    st.metric(label="Puissance Estimée", value=f"{max(0.0, pred):.2f} mW")
