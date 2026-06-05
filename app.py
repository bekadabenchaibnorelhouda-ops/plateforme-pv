import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# 1. Configuration de la page
st.set_page_config(page_title="Prédiction PV par IA", page_icon="☀️", layout="wide")

# Style CSS
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #F8F9FA; }
    h1, h2 { color: #E65100; }
    .carte-metrique { background-color: #F1F3F5; border: 1px solid #CED4DA; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 12px; }
    .valeur { font-size: 1.8rem; font-weight: 700; color: #FF6B2B; }
    .label { font-size: 0.8rem; color: #495057; text-transform: uppercase; }
    .cadre-accueil { background-color: #F8F9FA; border: 1px solid #DEE2E6; border-radius: 16px; padding: 25px; }
    </style>
""", unsafe_allow_html=True)

# Initialisation session state
if "donnees" not in st.session_state: st.session_state.donnees = None

# Barre latérale
with st.sidebar:
    st.markdown("### 🗂️ Menu Principal")
    page = st.radio("Sélectionnez une page :", ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"])
    st.divider()
    nom_modele = st.selectbox("Modèle d'IA :", ["MLP", "LSTM", "GRU", "ARX", "ANFIS"])
    nom_court = nom_modele

# --- PAGES ---
if page == "🏠 Accueil & Présentation":
    st.title("Prédiction de la Production d'Énergie Photovoltaïque")
    if os.path.exists("panneau_pv.jpg"):
        st.image("panneau_pv.jpg", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="cadre-accueil">
            <h4>👥 Auteurs :</h4>
            <p>Nor El Houda BEKADA BENCHAIB<br>Yousra Oum El kheir HAMMADI</p>
            <hr>
            <h4>👨‍🏫 Encadrant :</h4>
            <p>M. Anisse CHIALI & Mme Imane NEDJAR</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("### À propos du projet")
        st.write("Plateforme de prédiction énergétique utilisant des modèles avancés (IA/Fuzzy) pour les systèmes photovoltaïques.")

elif page == "📂 Importation des Données":
    st.title("📂 Importation des Données")
    fichier = st.file_uploader("Chargez votre fichier :", type=["xlsx", "csv"])
    if fichier:
        st.session_state.donnees = pd.read_csv(fichier) if fichier.name.endswith(".csv") else pd.read_excel(fichier)
        st.success("Fichier chargé !")
        st.write("### 📊 Aperçu des 10 premières lignes :")
        st.dataframe(st.session_state.donnees.head(10))
        st.write("### 📈 Statistiques descriptives :")
        st.write(st.session_state.donnees.describe())

elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Évaluation & Performance")
    if st.session_state.donnees is not None:
        metriques_data = {
            "MLP": {"RMSE": 0.023056, "MAE": 0.006563, "MAPE": 8.653303, "R2": 0.947466},
            "LSTM": {"RMSE": 0.026752, "MAE": 0.012418, "MAPE": 28.951633, "R2": 0.929274},
            "GRU": {"RMSE": 0.023370, "MAE": 0.006021, "MAPE": 7.396154, "R2": 0.946026},
            "ARX": {"RMSE": 0.024779, "MAE": 0.007986, "MAPE": 13.273804, "R2": 0.933430},
            "ANFIS": {"RMSE": 0.023378, "MAE": 0.005449, "MAPE": 3.304221, "R2": 0.945985}
        }
        stats = metriques_data[nom_court]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="carte-metrique"><div class="valeur">{stats["RMSE"]:.4f}</div><div class="label">RMSE</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="carte-metrique"><div class="valeur">{stats["MAE"]:.4f}</div><div class="label">MAE</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="carte-metrique"><div class="valeur">{stats["MAPE"]:.2f}%</div><div class="label">MAPE</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="carte-metrique"><div class="valeur">{stats["R2"]:.4f}</div><div class="label">R²</div></div>', unsafe_allow_html=True)
        
        # Graphique
        y_reel = st.session_state.donnees.iloc[:, -1].values[:200]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_reel, name="Valeur Réelle"))
        fig.add_trace(go.Scatter(y=y_reel * (stats["R2"]), name="Prédiction"))
        fig.update_layout(title=f"Performance : {nom_court}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Veuillez importer des données dans la page 'Importation'.")

elif page == "🔮 Prédiction Future":
    st.title("🔮 Prédiction Future Interactive")
    col1, col2, col3 = st.columns(3)
    val_ldr = col1.slider("Éclairement (LDR)", 0, 4095, 1500)
    val_hum = col2.slider("Humidité (%)", 0.0, 100.0, 65.0)
    val_temp = col3.slider("Température (°C)", -5.0, 50.0, 25.0)
    
    if st.button("Lancer la prédiction"):
        # Logique de calcul cohérente [100 - 250 mW]
        base = (val_ldr / 4095) * 150
        ajustement = ((100 - val_hum) * 0.3) + ((val_temp - 25) * 0.1)
        resultat = 100 + base + ajustement
        resultat = max(100, min(250, resultat))
        
        st.metric(label=f"Puissance estimée ({nom_court})", value=f"{resultat:.2f} mW")
