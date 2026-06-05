import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Configuration
st.set_page_config(page_title="Prédiction PV par IA", layout="wide")

# 2. Chargement sécurisé des ressources
@st.cache_resource
def charger_ressources():
    modeles = {}
    scaler = None
    
    # Chargement du Scaler avec vérification de taille
    if os.path.exists("scaler.pkl") and os.path.getsize("scaler.pkl") > 10:
        scaler = joblib.load("scaler.pkl")
    
    # Chargement modèles .pkl
    for m in ["mlp", "arx", "anfis"]:
        chemin = f"model_{m}.pkl"
        if os.path.exists(chemin) and os.path.getsize(chemin) > 10:
            modeles[m] = joblib.load(chemin)
            
    # Chargement modèles .h5
    try:
        from tensorflow.keras.models import load_model
        for m in ["lstm", "gru"]:
            chemin = f"model_{m}.h5"
            if os.path.exists(chemin) and os.path.getsize(chemin) > 10:
                modeles[m] = load_model(chemin, compile=False)
    except: pass
    return modeles, scaler

modeles, scaler = charger_ressources()

# --- NAVIGATION ---
st.sidebar.title("☀️ Menu Principal")
page = st.sidebar.radio("Navigation :", ["🏠 Accueil", "📂 Importation", "📊 Évaluation", "🔮 Prédiction"])
modele_choisi = st.sidebar.selectbox("Choisir le Modèle :", ["LSTM", "GRU", "MLP", "ARX", "ANFIS"])
cle_modele = modele_choisi.lower()

# --- PAGES ---
if page == "🏠 Accueil":
    st.title("Projet de Fin d'Études")
    st.write("Auteurs : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI")
    st.write("Encadrement : M. Anisse CHIALI & Mme Imane NEDJAR")
    if os.path.exists("panneau_pv.jpg"): st.image("panneau_pv.jpg", use_container_width=True)

elif page == "📂 Importation":
    st.title("📂 Importation des Données")
    file = st.file_uploader("Charger fichier CSV/Excel", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        st.session_state["donnees"] = df
        st.success("Données chargées !")

elif page == "📊 Évaluation":
    st.title("📊 Évaluation des Modèles")
    if "donnees" not in st.session_state: st.warning("Chargez des données."); st.stop()
    if cle_modele not in modeles: st.error(f"Le modèle {modele_choisi} n'est pas chargé (vérifiez vos fichiers)."); st.stop()
    
    df = st.session_state["donnees"].dropna()
    X = df[["LDR_Raw", "Hum_%", "Temp_C"]].values
    if scaler: X = scaler.transform(X)
    y_reel = df["Puissance_mW"].values
    
    model = modeles[cle_modele]
    if cle_modele in ["lstm", "gru"]: X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    y_pred = np.clip(model.predict(X).flatten(), 0, None)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_reel, y_pred)):.2f}")
    col2.metric("MAE", f"{mean_absolute_error(y_reel, y_pred):.2f}")
    col3.metric("R²", f"{r2_score(y_reel, y_pred):.4f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_reel[:100], name="Réel"))
    fig.add_trace(go.Scatter(y=y_pred[:100], name="Prédit"))
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction Interactive")
    ldr = st.slider("Éclairement", 0, 2000, 500)
    hum = st.slider("Humidité", 0, 100, 50)
    tmp = st.slider("Température", -10, 50, 25)
    
    if st.button("Lancer la Prédiction"):
        if cle_modele not in modeles: st.error("Modèle introuvable."); st.stop()
        x = np.array([[ldr, hum, tmp]])
        if scaler: x = scaler.transform(x)
        if cle_modele in ["lstm", "gru"]: x = x.reshape((1, 1, 3))
        
        pred = modeles[cle_modele].predict(x).flatten()[0]
        st.metric("Puissance Prévue", f"{max(0.0, pred):.2f} mW")
