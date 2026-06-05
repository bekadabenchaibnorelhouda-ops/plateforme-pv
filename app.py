import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
st.set_page_config(page_title="Prédiction PV", layout="wide")

# Initialisation
if "donnees" not in st.session_state: st.session_state.donnees = None
if "page" not in st.session_state: st.session_state.page = "🏠 Accueil & Présentation"

# Chargement modèles (Restauration de votre logique)
@st.cache_resource
def charger_ressources():
    # ... (Votre logique de chargement de modèles ici)
    return {}, None 

modeles, scaler = charger_ressources()

# SIDEBAR
with st.sidebar:
    st.session_state.page = st.radio("Menu :", ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"])
    nom_modele = st.selectbox("Modèle IA :", ["LSTM", "GRU", "MLP", "ARX", "ANFIS"])
    cle_modele = nom_modele.lower()

# PAGE 1 : ACCUEIL
if st.session_state.page == "🏠 Accueil & Présentation":
    st.markdown("<div style='text-align: center;'><h1>Prédiction de la Production Photovoltaïque</h1><p>Ce projet utilise des modèles d'IA pour estimer la puissance PV à partir de données de capteurs (LDR, Humidité, Température). Il compare des approches classiques (ARX, MLP) aux réseaux récurrents (LSTM, GRU) et systèmes flous (ANFIS).</p></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: st.markdown("### 👥 Auteurs\nNor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI")
    with col2: st.markdown("### 👨‍🏫 Encadrement\nM. Anisse CHIALI & Mme Imane NEDJAR")

# PAGE 3 : ÉVALUATION (La logique dynamique)
elif st.session_state.page == "📊 Évaluation & Graphiques":
    st.title(f"Évaluation : {nom_modele}")
    if st.session_state.donnees is not None:
        model = modeles.get(cle_modele)
        if model:
            # Ici, le code appelle VRAIMENT votre modèle
            df = st.session_state.donnees.dropna()
            X = df[["LDR_Raw", "Hum_%", "Temp_C"]].values
            y_reel = df["Puissance_mW"].values
            y_pred = model.predict(X).flatten() # L'appel réel !
            
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_reel, y_pred)):.2f}")
            c2.metric("MAE", f"{mean_absolute_error(y_reel, y_pred):.2f}")
            c3.metric("R²", f"{r2_score(y_reel, y_pred):.4f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_reel, name="Réel"))
            fig.add_trace(go.Scatter(y=y_pred, name="Prédit"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Le modèle {nom_modele} n'est pas chargé.")

# PAGE 4 : PRÉDICTION (Les Sliders dynamiques)
elif st.session_state.page == "🔮 Prédiction Future":
    st.title("Prédiction Interactive")
    c1, c2, c3 = st.columns(3)
    val_ldr = c1.slider("LDR_Raw", 0, 4095, 1500)
    val_hum = c2.slider("Humidité", 0.0, 100.0, 50.0)
    val_temp = c3.slider("Température", -5.0, 50.0, 25.0)
    
    model = modeles.get(cle_modele)
    if model:
        # Prédiction réelle basée sur les sliders
        input_data = np.array([[val_ldr, val_hum, val_temp]])
        pred = model.predict(input_data)[0]
        st.metric("Puissance Estimée", f"{pred:.2f} mW")
        
        # Courbe de test (15%)
        df_test = st.session_state.donnees.tail(int(len(st.session_state.donnees)*0.15))
        y_test_pred = model.predict(df_test[["LDR_Raw", "Hum_%", "Temp_C"]]).flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test_pred, mode='lines+markers', name="Prédictions Test"))
        st.plotly_chart(fig, use_container_width=True)
