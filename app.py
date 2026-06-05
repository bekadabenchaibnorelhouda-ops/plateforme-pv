import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Prédiction PV par IA", page_icon="☀️", layout="wide")

# --- CSS PERSONNALISÉ (Votre structure visuelle) ---
st.markdown("""
    <style>
    :root { --couleur-primaire: #FF6B2B; --couleur-secondaire: #F8F9FA; --couleur-accent: #E65100; --couleur-fond: #FFFFFF; --couleur-carte: #F1F3F5; }
    .stApp { background-color: var(--couleur-fond); }
    .carte-metrique { background-color: var(--couleur-carte); border: 1px solid #CED4DA; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .carte-metrique .valeur { font-size: 2rem; font-weight: 700; color: var(--couleur-primaire); }
    .cadre-accueil { background-color: #F8F9FA; border: 1px solid #DEE2E6; border-radius: 16px; padding: 25px; }
    .badge-pfe { background-color: #E65100; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- PARAMÈTRES ---
COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C"]
COLONNE_CIBLE = "Puissance_mW"
MODELES_DISPONIBLES = {
    "💾 LSTM": "lstm", "🔁 GRU": "gru", "🧠 MLP": "mlp", "📐 ARX": "arx", "🔮 ANFIS": "anfis"
}

@st.cache_resource
def charger_ressources():
    modeles = {}
    scaler = None
    if os.path.exists("scaler.pkl"): scaler = joblib.load("scaler.pkl")
    for cle in ["mlp", "arx", "anfis"]:
        if os.path.exists(f"model_{cle}.pkl"): modeles[cle] = joblib.load(f"model_{cle}.pkl")
    try:
        from tensorflow.keras.models import load_model
        for cle in ["lstm", "gru"]:
            if os.path.exists(f"model_{cle}.h5"): modeles[cle] = load_model(f"model_{cle}.h5", compile=False)
    except: pass
    return modeles, scaler

modeles, scaler = charger_ressources()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🗂️ Menu Principal")
    page = st.radio("Navigation :", ["🏠 Accueil", "📂 Importation", "📊 Évaluation", "🔮 Prédiction"])
    st.divider()
    nom_modele_selectionne = st.selectbox("Modèle actif :", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele_selectionne]

# --- PAGES ---
if page == "🏠 Accueil":
    st.markdown("<div style='text-align: center;'><span class='badge-pfe'>PROJET DE FIN D'ÉTUDES</span><h1>Prédiction PV par IA</h1></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='cadre-accueil'><h4>👥 Auteurs :</h4> Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI<br><br><h4>👨‍🏫 Encadrement :</h4> M. Anisse CHIALI & Mme Imane NEDJAR</div>", unsafe_allow_html=True)
    with col2:
        st.write("Plateforme d'analyse et de prédiction énergétique basée sur le traitement de données capteurs.")

elif page == "📂 Importation":
    st.title("📂 Importation")
    file = st.file_uploader("Charger CSV/Excel", type=["csv", "xlsx"])
    if file:
        st.session_state.donnees = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        st.success("Données chargées !")
    if "donnees" in st.session_state and st.session_state.donnees is not None:
        st.dataframe(st.session_state.donnees.head())

elif page == "📊 Évaluation":
    st.title("📊 Évaluation")
    if "donnees" not in st.session_state or st.session_state.donnees is None: st.warning("Chargez des données."); st.stop()
    
    df = st.session_state.donnees.dropna()
    # Logique robuste de sélection de features
    cols_a_utiliser = ["LDR_Raw"] if cle_modele in ["arx", "anfis"] else COLONNES_REQUISES
    X = df[cols_a_utiliser].values.astype(float)
    
    if scaler:
        # Vérification de dimension pour éviter le ValueError
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == X.shape[1]:
            X = scaler.transform(X)
            
    obj_modele = modeles.get(cle_modele)
    if cle_modele in ["gru", "lstm"]: X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    y_pred = np.clip(obj_modele.predict(X).flatten(), 0, None)
    y_reel = df[COLONNE_CIBLE].values
    
    # Métriques
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='carte-metrique'><div class='valeur'>{np.sqrt(mean_squared_error(y_reel, y_pred)):.2f}</div><div class='label'>RMSE</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='carte-metrique'><div class='valeur'>{mean_absolute_error(y_reel, y_pred):.2f}</div><div class='label'>MAE</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='carte-metrique'><div class='valeur'>{r2_score(y_reel, y_pred):.4f}</div><div class='label'>R²</div></div>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_reel[:200], name="Réel"))
    fig.add_trace(go.Scatter(y=y_pred[:200], name="Prédit"))
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction")
    val_ldr = st.slider("LDR", 0, 4095, 1500)
    if cle_modele in ["mlp", "lstm", "gru"]:
        val_hum = st.slider("Hum", 0.0, 100.0, 50.0)
        val_temp = st.slider("Temp", -5.0, 50.0, 25.0)
        X = np.array([[val_ldr, val_hum, val_temp]])
    else:
        X = np.array([[val_ldr]])
        
    if scaler and hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == X.shape[1]:
        X = scaler.transform(X)
        
    if cle_modele in ["lstm", "gru"]: X = X.reshape((1, 1, X.shape[1]))
    pred = modeles[cle_modele].predict(X).flatten()[0]
    st.metric("Puissance Estimée", f"{max(0.0, pred):.2f} mW")
