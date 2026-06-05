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

# Style CSS personnalisé pour l'interface utilisateur
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
COLONNE_CIBLE     = "Puissance_mW"
FICHIER_EXEMPLE   = "Classeur1.xlsx"

MODELES_DISPONIBLES = {
    "💾 LSTM (Long Short-Term Memory)"         : "lstm",
    "🔁 GRU (Gated Recurrent Unit)"            : "gru",
    "🧠 MLP (Multi-Layer Perceptron)"          : "mlp",
    "📐 ARX (Auto-Regressive Exogenous)"       : "arx",
    "🔮 ANFIS (Adaptive Neuro-Fuzzy)"          : "anfis",
}

@st.cache_resource(show_spinner="⚙️ Chargement des architectures d'IA…")
def charger_ressources():
    modeles = {}
    scaler = None
    if os.path.exists("scaler.pkl"):
        try: scaler = joblib.load("scaler.pkl")
        except: pass
    fichiers_pkl = {"mlp": "model_mlp.pkl", "arx": "model_arx.pkl", "anfis": "model_anfis.pkl"}
    for cle, chemin in fichiers_pkl.items():
        if os.path.exists(chemin):
            try: modeles[cle] = joblib.load(chemin)
            except: pass
    try:
        from tensorflow.keras.models import load_model
        fichiers_h5 = {"gru": "model_gru.h5", "lstm": "model_lstm.h5"}
        for cle, chemin in fichiers_h5.items():
            if os.path.exists(chemin):
                try: modeles[cle] = load_model(chemin, compile=False)
                except: pass
    except: pass
    return modeles, scaler

modeles, scaler = charger_ressources()

if "donnees" not in st.session_state: st.session_state.donnees = None
if "nom_fichier" not in st.session_state: st.session_state.nom_fichier = None

with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 10px 0;'><div style='font-size:3rem;'>☀️</div></div>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Menu Principal")
    page = st.radio("Sélectionnez une page :", ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"])
    st.divider()
    st.markdown("### 🤖 Configuration IA")
    nom_modele_selectionne = st.selectbox("Modèle d'IA actif :", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele_selectionne]
    nom_court = nom_modele_selectionne.split("(")[0].strip()

# --- PAGE 1 ---
if page == "🏠 Accueil & Présentation":
    st.markdown("""<div style="text-align: center; margin-top: 25px;"><span class="badge-pfe">PROJET DE FIN D'ÉTUDES (PFE)</span><h1>Prédiction de la Production d'Énergie PV</h1></div>""", unsafe_allow_html=True)

# --- PAGE 2 ---
elif page == "📂 Importation des Données":
    st.title("📂 Importation des Données")
    fichier_charge = st.file_uploader("Choisir un fichier :", type=["xlsx", "csv"])
    if fichier_charge:
        df = pd.read_csv(fichier_charge) if fichier_charge.name.endswith(".csv") else pd.read_excel(fichier_charge)
        st.session_state.donnees = df
        st.dataframe(df.head())
        # Ajout du tableau statistique
        st.subheader("📊 Statistiques des données")
        st.table(df.describe().loc[['min', 'max', 'mean', 'std']])

# --- PAGE 3 ---
elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Évaluation")
    if st.session_state.donnees is not None:
        df = st.session_state.donnees.dropna()
        
        # CORRECTION : Sélection dynamique des colonnes selon le modèle
        if cle_modele in ["arx", "anfis"]:
            X = df[["LDR_Raw"]].values
        else:
            X = df[COLONNES_REQUISES].values
            
        y_reel = df[COLONNE_CIBLE].values
        
        # CORRECTION : On vérifie si le scaler accepte bien X avant de transformer
        try:
            X_scaled = scaler.transform(X) if scaler else X
            obj_modele = modeles.get(cle_modele)
            y_pred = obj_modele.predict(X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]) if cle_modele in ["gru", "lstm"] else X_scaled).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
            st.metric("RMSE", f"{rmse:.2f}")
        except Exception as e:
            st.error(f"Erreur de compatibilité modèle/données : {e}")

# --- PAGE 4 ---
elif page == "🔮 Prédiction Future":
    st.title("🔮 Prédiction")
