"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PLATEFORME DE PRÉDICTION D'ÉNERGIE PHOTOVOLTAÏQUE PAR IA          ║
║                Projet de Fin d'Études — Ingénierie des Systèmes            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Auteurs     : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI
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
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(page_title="Prédiction PV par IA", page_icon="☀️", layout="wide")

# (Le bloc CSS reste exactement comme le vôtre)
st.markdown("""
    <style>
    .carte-metrique { background-color: #F1F3F5; border: 1px solid #CED4DA; border-radius: 12px; padding: 20px; text-align: center; }
    .valeur { font-size: 2rem; font-weight: 700; color: #FF6B2B; }
    .label { font-size: 0.85rem; color: #495057; text-transform: uppercase; }
    .cadre-accueil { background-color: #F8F9FA; border: 1px solid #DEE2E6; border-radius: 16px; padding: 25px; }
    .badge-pfe { background-color: #E65100; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ... (Logique de chargement charger_ressources et configuration inchangée)

# --- PAGE 1 : RESTAURATION ACCUEIL ---
if page == "🏠 Accueil & Présentation":
    if os.path.exists("panneau_pv.jpg"):
        st.image("panneau_pv.jpg", caption="Dispositif expérimental d'acquisition de données", use_container_width=True)
    st.markdown("""<div style="text-align: center;"><span class="badge-pfe">PROJET DE FIN D'ÉTUDES (PFE)</span><h1>Prédiction de la Production d'Énergie Photovoltaïque par Intelligence Artificielle</h1></div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📝 Fiche Technique du Projet")
        st.markdown("""<div class="cadre-accueil"><h4>👥 Réalisé par :</h4><ul><li><b>Nor El Houda BEKADA BENCHAIB</b></li><li><b>Yousra Oum El kheir HAMMADI</b></li></ul><hr/><h4>👨‍🏫 Encadré par :</h4><p><b>M. Anisse CHIALI</b></p><p><b>Mme Imane NEDJAR</b></p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("### 💡 À propos")
        st.write("Plateforme d'analyse et de prédiction basée sur l'IA.")

# --- PAGE 3 : ÉVALUATION (Découpage 70/15/15) ---
elif page == "📊 Évaluation & Graphiques":
    # Logique de split
    n = len(df_global)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    # Visualisation des 3 segments
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_reel, name="Réel"))
    fig.add_trace(go.Scatter(y=y_pred, name="Prédiction"))
    fig.add_vline(x=train_end, line_dash="dash", annotation_text="Fin Train")
    fig.add_vline(x=val_end, line_dash="dash", annotation_text="Fin Val")
    st.plotly_chart(fig)

# --- PAGE 4 : PRÉDICTION (Courbe unique de test) ---
elif page == "🔮 Prédiction Future":
    # Curseurs conservés
    val_ldr = st.slider("LDR_Raw", 0, 4095, 1500)
    # ... (autres sliders)
    
    # Courbe unique sans le réel
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(y=preds_test, name="Prédiction Test", line=dict(color="#FF6B2B")))
    fig_test.update_layout(title="Courbe de Test (Prédictions)")
    st.plotly_chart(fig_test)
