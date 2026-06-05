import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- CONFIGURATION ---
st.set_page_config(page_title="Prédiction PV par IA", layout="wide")

# Initialisation
if "donnees" not in st.session_state: st.session_state.donnees = None
if "page" not in st.session_state: st.session_state.page = "🏠 Accueil & Présentation"

# Chargement dynamique des modèles
@st.cache_resource(show_spinner="⚙️ Chargement des modèles...")
def charger_ressources():
    # Remplacez par votre logique de chargement existante
    modeles = {} 
    scaler = None 
    return modeles, scaler

modeles, scaler = charger_ressources()

# --- SIDEBAR ---
with st.sidebar:
    st.session_state.page = st.radio("Menu :", ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"])
    nom_modele_sel = st.selectbox("Modèle IA :", ["LSTM", "GRU", "MLP", "ARX", "ANFIS"])
    cle_modele = nom_modele_sel.lower()
    obj_modele = modeles.get(cle_modele) # Modèle actif sélectionné

# --- PAGE 1 : ACCUEIL ---
if st.session_state.page == "🏠 Accueil & Présentation":
    if os.path.exists("panneau_pv.jpg"): st.image("panneau_pv.jpg", use_container_width=True)
    st.title("Projet de Fin d'Études : Prédiction PV par IA")
    st.markdown("""
    ### À propos de cette plateforme
    Ce projet vise à optimiser la gestion de l'énergie photovoltaïque. En utilisant des capteurs (LDR, Humidité, Température), 
    nous déployons des modèles d'IA (Deep Learning et Neuro-Flou) pour anticiper la production électrique réelle.
    Cette application permet d'évaluer la précision de nos modèles sur des données expérimentales et de simuler 
    la production future en temps réel.
    """)
    c1, c2 = st.columns(2)
    c1.info("**Auteurs :** Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI")
    c2.info("**Encadrement :** M. Anisse CHIALI & Mme Imane NEDJAR")

# --- PAGE 3 : ÉVALUATION (Dynamique) ---
elif st.session_state.page == "📊 Évaluation & Graphiques":
    st.title(f"📊 Évaluation avec : {nom_modele_sel}")
    if st.session_state.donnees is not None and obj_modele is not None:
        # Calcul dynamique
        X = st.session_state.donnees.iloc[:, :-1].values
        y_reel = st.session_state.donnees.iloc[:, -1].values
        y_pred = obj_modele.predict(X) # Utilisation réelle du modèle sélectionné
        
        # Métriques dynamiques
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_reel, y_pred)):.4f}")
        c2.metric("MAE", f"{mean_absolute_error(y_reel, y_pred):.4f}")
        c3.metric("R²", f"{r2_score(y_reel, y_pred):.4f}")
        
        # Graphique réel vs prédit
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_reel, name="Réel", line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=y_pred, name="Prédit", line=dict(color='red')))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Modèle introuvable ou données manquantes.")

# --- PAGE 4 : PRÉDICTION (Interactive) ---
elif st.session_state.page == "🔮 Prédiction Future":
    st.title("🔮 Prédiction")
    # Curseurs dynamiques
    l = st.slider("LDR", 0, 4095, 1500)
    h = st.slider("Humidité", 0.0, 100.0, 50.0)
    t = st.slider("Temp", -5.0, 50.0, 25.0)
    
    if obj_modele:
        # Inférence réelle
        input_data = np.array([[l, h, t]])
        result = obj_modele.predict(input_data)[0]
        st.metric("Puissance Estimée", f"{result:.2f} mW")
        
        # Courbe de Test (15% derniers points)
        st.subheader("Visualisation de la performance (Test 15%)")
        test_data = st.session_state.donnees.iloc[-int(len(st.session_state.donnees)*0.15):]
        preds_test = obj_modele.predict(test_data.iloc[:, :-1])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=preds_test, mode='lines+markers', name="Prédictions Test"))
        st.plotly_chart(fig, use_container_width=True)
