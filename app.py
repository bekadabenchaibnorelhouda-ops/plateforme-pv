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

# Style CSS personnalisé
st.markdown(
    """
    <style>
    :root { --couleur-primaire: #FF6B2B; --couleur-secondaire: #F8F9FA; --couleur-accent: #E65100; --couleur-fond: #FFFFFF; --couleur-carte: #F1F3F5; --couleur-texte: #212529; }
    .stApp { background-color: var(--couleur-fond); color: var(--couleur-texte); font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: var(--couleur-secondaire); border-right: 1px solid #DEE2E6; }
    h1, h2, h3 { color: var(--couleur-accent) !important; font-weight: 700 !important; }
    .carte-metrique { background-color: var(--couleur-carte); border: 1px solid #CED4DA; border-radius: 12px; padding: 20px 24px; text-align: center; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); margin-bottom: 12px; }
    .carte-metrique .valeur { font-size: 2rem; font-weight: 700; color: var(--couleur-primaire); }
    .carte-metrique .label { font-size: 0.85rem; color: #495057; margin-top: 4px; text-transform: uppercase; }
    .cadre-accueil { background-color: #F8F9FA; border: 1px solid #DEE2E6; border-radius: 16px; padding: 25px; margin-bottom: 20px; }
    .badge-pfe { background-color: #E65100; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; display: inline-block; margin-bottom: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C"]
COLONNE_CIBLE     = "Puissance_mW"
MODELES_DISPONIBLES = {
    "💾 LSTM (Long Short-Term Memory)": "lstm",
    "🔁 GRU (Gated Recurrent Unit)": "gru",
    "🧠 MLP (Multi-Layer Perceptron)": "mlp",
    "📐 ARX (Auto-Regressive Exogenous)": "arx",
    "🔮 ANFIS (Adaptive Neuro-Fuzzy)": "anfis",
}

@st.cache_resource
def charger_ressources():
    modeles = {}
    scaler = None
    if os.path.exists("scaler.pkl"): scaler = joblib.load("scaler.pkl")
    fichiers = {"mlp": "model_mlp.pkl", "arx": "model_arx.pkl", "anfis": "model_anfis.pkl"}
    for k, c in fichiers.items():
        if os.path.exists(c): modeles[k] = joblib.load(c)
    try:
        from tensorflow.keras.models import load_model
        for k in ["gru", "lstm"]:
            if os.path.exists(f"model_{k}.h5"): modeles[k] = load_model(f"model_{k}.h5", compile=False)
    except: pass
    return modeles, scaler

modeles, scaler = charger_ressources()

if "donnees" not in st.session_state: st.session_state.donnees = None

with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 10px 0;'><div style='font-size:3rem;'>☀️</div></div>", unsafe_allow_html=True)
    page = st.radio("Sélectionnez une page :", ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"])
    nom_modele_selectionne = st.selectbox("Modèle d'IA actif :", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele_selectionne]

if page == "🏠 Accueil & Présentation":
    st.markdown("""<div style="text-align: center;"><span class="badge-pfe">PROJET DE FIN D'ÉTUDES (PFE)</span><h1>Prédiction PV par IA</h1></div>""", unsafe_allow_html=True)

elif page == "📂 Importation des Données":
    st.title("📂 Importation des Données")
    f = st.file_uploader("Charger fichier :", type=["xlsx", "csv"])
    if f:
        st.session_state.donnees = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
    
    if st.session_state.donnees is not None:
        st.write("### Aperçu")
        st.dataframe(st.session_state.donnees.head())
        st.write("### Statistiques (Min, Max, Moyenne)")
        # Ajout du tableau de statistiques demandé
        stats = st.session_state.donnees.describe().loc[['min', 'max', 'mean', 'std']]
        st.table(stats)

elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Évaluation")
    if st.session_state.donnees is None: st.stop()
    
    df = st.session_state.donnees.dropna(subset=COLONNES_REQUISES + [COLONNE_CIBLE])
    y_reel = df[COLONNE_CIBLE].values
    
    # CORRECTION DE L'ERREUR : Sélection dynamique des colonnes selon le modèle
    colonnes_utilisees = ["LDR_Raw"] if cle_modele in ["arx", "anfis"] else COLONNES_REQUISES
    X = df[colonnes_utilisees].values
    
    # Transformation sécurisée
    if scaler and hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == X.shape[1]:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X # Fallback si scaler incompatible
        
    obj_modele = modeles.get(cle_modele)
    if cle_modele in ["gru", "lstm"]:
        y_pred = obj_modele.predict(X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]), verbose=0).flatten()
    else:
        y_pred = obj_modele.predict(X_scaled).flatten()
        
    st.metric("R² Score", f"{r2_score(y_reel, y_pred):.4f}")
    # ... (reste de votre code d'affichage Plotly)
