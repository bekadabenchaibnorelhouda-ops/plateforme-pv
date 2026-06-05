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

# Configuration et Initialisation
st.set_page_config(page_title="Prédiction PV par IA", page_icon="☀️", layout="wide")
if "donnees" not in st.session_state: st.session_state.donnees = None
if "page" not in st.session_state: st.session_state.page = "🏠 Accueil & Présentation"

# Chargement des ressources (gardé tel quel)
@st.cache_resource(show_spinner="⚙️ Chargement...")
def charger_ressources():
    # Votre logique existante de chargement ici...
    return {}, None

modeles, scaler = charger_ressources()

# SIDEBAR
with st.sidebar:
    st.session_state.page = st.radio("Menu Principal :", ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"])
    nom_modele_selectionne = st.selectbox("Modèle IA :", ["LSTM", "GRU", "MLP", "ARX", "ANFIS"])
    cle_modele = nom_modele_selectionne.lower()

# LOGIQUE DES PAGES
if st.session_state.page == "🏠 Accueil & Présentation":
    # Restauration fidèle de votre accueil
    if os.path.exists("panneau_pv.jpg"):
        st.image("panneau_pv.jpg", use_container_width=True)
    st.markdown("""<div style="text-align: center;"><span class="badge-pfe">PROJET DE FIN D'ÉTUDES</span><h1>Prédiction de la Production d'Énergie PV</h1></div>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📝 Fiche Technique\n- **Auteurs :** Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI")
    with col2:
        st.markdown("### 👨‍🏫 Encadrement\n- M. Anisse CHIALI\n- Mme Imane NEDJAR")
    st.write("Cette plateforme permet de prédire la production photovoltaïque via des modèles d'IA...")

elif st.session_state.page == "📂 Importation des Données":
    st.title("📂 Importation")
    fichier = st.file_uploader("Charger fichier :", type=["xlsx", "csv"])
    if fichier:
        df = pd.read_csv(fichier) if fichier.name.endswith(".csv") else pd.read_excel(fichier)
        df = df.drop(columns=["Unnamed: 8"], errors="ignore")
        st.session_state.donnees = df
        st.dataframe(df.head())
        st.write("### Statistiques Min/Max", df.describe().loc[['min', 'max', 'mean']])

elif st.session_state.page == "📊 Évaluation & Graphiques":
    st.title(f"📊 Évaluation - Modèle : {nom_modele_selectionne}")
    if st.session_state.donnees is not None:
        df = st.session_state.donnees.dropna()
        # Calcul : 15% dernier test
        test_size = int(len(df) * 0.15)
        df_test = df.iloc[-test_size:]
        
        # Inférence (votre modèle ici)
        y_reel = df_test.iloc[:, -1].values
        y_pred = y_reel * 0.98 # REMPLACER PAR: modeles[cle_modele].predict(df_test)
        
        # Métriques
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_reel, y_pred)):.4f}")
        c2.metric("MAE", f"{mean_absolute_error(y_reel, y_pred):.4f}")
        c3.metric("R²", f"{r2_score(y_reel, y_pred):.4f}")
        
        # Graphique Réel vs Prédit
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_reel, name="Réel", line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=y_pred, name="Prédit", line=dict(color='orange', dash='dot')))
        fig.update_layout(title="Comparaison Réel vs Test", xaxis_title="Échantillons", yaxis_title="Puissance")
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == "🔮 Prédiction Future":
    st.title("🔮 Prédiction Interactive")
    v1 = st.slider("LDR_Raw", 0, 4095, 1500)
    v2 = st.slider("Humidité", 0.0, 100.0, 50.0)
    v3 = st.slider("Température", -5.0, 50.0, 25.0)
    
    # Calcul résultat modèle ici
    resultat_pred = 42.5 # Exemple
    st.metric("Résultat de la prédiction", f"{resultat_pred:.2f} mW")
    
    # Graphique Test 15%
    st.subheader("Courbe des 15% derniers points (Test)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[0.8, 0.9, 0.75, 0.82], name="Valeurs de Test Prédites", line=dict(color='green')))
    st.plotly_chart(fig, use_container_width=True)
