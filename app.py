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

# Style CSS conservé
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
    "💾 LSTM (Long Short-Term Memory)": "lstm",
    "🔁 GRU (Gated Recurrent Unit)": "gru",
    "🧠 MLP (Multi-Layer Perceptron)": "mlp",
    "📐 ARX (Auto-Regressive Exogenous)": "arx",
    "🔮 ANFIS (Adaptive Neuro-Fuzzy)": "anfis",
}

@st.cache_resource(show_spinner="⚙️ Chargement des architectures d'IA…")
def charger_ressources():
    modeles = {}
    scaler = None
    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
    
    fichiers_pkl = {"mlp": "model_mlp.pkl", "arx": "model_arx.pkl", "anfis": "model_anfis.pkl"}
    for cle, chemin in fichiers_pkl.items():
        if os.path.exists(chemin): modeles[cle] = joblib.load(chemin)
    
    try:
        from tensorflow.keras.models import load_model
        if os.path.exists("model_gru.h5"): modeles["gru"] = load_model("model_gru.h5", compile=False)
        if os.path.exists("model_lstm.h5"): modeles["lstm"] = load_model("model_lstm.h5", compile=False)
    except: pass
    return modeles, scaler

modeles, scaler = charger_ressources()

if "donnees" not in st.session_state: st.session_state.donnees = None
if "nom_fichier" not in st.session_state: st.session_state.nom_fichier = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 10px 0;'><div style='font-size:3rem;'>☀️</div></div>", unsafe_allow_html=True)
    page = st.radio("Sélectionnez une page :", ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"])
    nom_modele_selectionne = st.selectbox("Modèle d'IA actif :", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele_selectionne]

# --- LOGIQUE DE PRÉDICTION SÉCURISÉE ---
def preparer_et_predire(df, modele, cle, scaler):
    X = df[COLONNES_REQUISES].values
    # Application rigoureuse du scaler (transform uniquement)
    if scaler is not None:
        X = scaler.transform(X)
    
    if cle in ["gru", "lstm"]:
        X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    preds = modele.predict(X, verbose=0).flatten()
    return np.maximum(0, preds) # Empêche les puissances négatives

# --- PAGE 3 CORRIGÉE ---
if page == "📊 Évaluation & Graphiques":
    st.title("📊 Évaluation Réelle")
    if st.session_state.donnees is not None:
        df = st.session_state.donnees.dropna(subset=COLONNES_REQUISES + [COLONNE_CIBLE])
        y_reel = df[COLONNE_CIBLE].values
        
        # Inférence corrigée
        y_pred = preparer_et_predire(df, modeles[cle_modele], cle_modele, scaler)
        
        # Calcul des métriques sur données non transformées (inverses)
        rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
        mae = mean_absolute_error(y_reel, y_pred)
        r2 = r2_score(y_reel, y_pred)
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="carte-metrique"><div class="valeur">{rmse:.2f}</div><div class="label">RMSE</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="carte-metrique"><div class="valeur">{mae:.2f}</div><div class="label">MAE</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="carte-metrique"><div class="valeur">{r2:.4f}</div><div class="label">R²</div></div>', unsafe_allow_html=True)
        
        # Graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_reel[:200], name="Réel", line=dict(color="#1A73E8")))
        fig.add_trace(go.Scatter(y=y_pred[:200], name="Prédit", line=dict(color="#FF6B2B", dash="dash")))
        st.plotly_chart(fig, use_container_width=True)
