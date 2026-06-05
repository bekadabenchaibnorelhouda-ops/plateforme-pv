import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# 1. Configuration de la page
st.set_page_config(page_title="Prédiction PV par IA", page_icon="☀️", layout="wide")

# CSS personnalisé (Gardé tel quel pour votre esthétique)
st.markdown("""
    <style>
    .carte-metrique { background-color: #F1F3F5; border: 1px solid #CED4DA; border-radius: 12px; padding: 20px; text-align: center; }
    .valeur { font-size: 1.8rem; font-weight: 700; color: #FF6B2B; }
    </style>
""", unsafe_allow_html=True)

# 2. Ressources (Chargement)
@st.cache_resource
def charger_ressources():
    modeles = {}
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    
    # Charger modèles .pkl
    for m in ["mlp", "arx", "anfis"]:
        if os.path.exists(f"model_{m}.pkl"): modeles[m] = joblib.load(f"model_{m}.pkl")
    
    # Charger modèles .h5
    try:
        from tensorflow.keras.models import load_model
        for m in ["lstm", "gru"]:
            if os.path.exists(f"model_{m}.h5"): modeles[m] = load_model(f"model_{m}.h5", compile=False)
    except: pass
    return modeles, scaler

modeles, scaler = charger_ressources()

# --- NAVIGATION ---
st.sidebar.title("☀️ Menu Principal")
page = st.sidebar.radio("Navigation :", ["🏠 Accueil", "📂 Importation", "📊 Évaluation", "🔮 Prédiction"])
modele_choisi = st.sidebar.selectbox("Modèle IA :", ["LSTM", "GRU", "MLP", "ARX", "ANFIS"])
cle_modele = modele_choisi.lower()

# --- PAGES ---
if page == "🏠 Accueil":
    st.title("Projet de Fin d'Études")
    st.write("Auteurs : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI")
    st.write("Encadrement : M. Anisse CHIALI & Mme Imane NEDJAR")
    if os.path.exists("panneau_pv.jpg"): st.image("panneau_pv.jpg", use_container_width=True)

elif page == "📂 Importation":
    st.title("📂 Données")
    file = st.file_uploader("Fichier CSV/Excel", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        st.session_state["donnees"] = df
        st.dataframe(df.head())

elif page == "📊 Évaluation":
    st.title("📊 Évaluation des Modèles")
    if "donnees" not in st.session_state: st.warning("Chargez des données d'abord."); st.stop()
    
    df = st.session_state["donnees"].dropna()
    X = scaler.transform(df[["LDR_Raw", "Hum_%", "Temp_C"]]) if scaler else df[["LDR_Raw", "Hum_%", "Temp_C"]].values
    y_reel = df["Puissance_mW"].values
    
    model = modeles.get(cle_modele)
    if cle_modele in ["lstm", "gru"]: X = X.reshape((X.shape[0], 1, X.shape[1]))
    y_pred = np.clip(model.predict(X).flatten(), 0, None)
    
    # Métriques corrigées
    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="carte-metrique"><div class="valeur">{np.sqrt(mean_squared_error(y_reel, y_pred)):.2f}</div>RMSE</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="carte-metrique"><div class="valeur">{mean_absolute_error(y_reel, y_pred):.2f}</div>MAE</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="carte-metrique"><div class="valeur">{r2_score(y_reel, y_pred):.4f}</div>R²</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_reel[:100], name="Réel"))
    fig.add_trace(go.Scatter(y=y_pred[:100], name="Prédit"))
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction Interactive")
    ldr = st.slider("LDR", 0, 2000, 500)
    hum = st.slider("Humidité", 0, 100, 50)
    tmp = st.slider("Température", -10, 50, 25)
    
    if st.button("Prédire"):
        x = np.array([[ldr, hum, tmp]])
        if scaler: x = scaler.transform(x)
        if cle_modele in ["lstm", "gru"]: x = x.reshape((1, 1, 3))
        
        pred = modeles[cle_modele].predict(x).flatten()[0]
        st.metric("Puissance Prévue", f"{max(0.0, pred):.2f} mW")
