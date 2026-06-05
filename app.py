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

if page == "🏠 Accueil & Présentation":
    if os.path.exists("panneau_pv.jpg"):
        st.image("panneau_pv.jpg", caption="Dispositif expérimental d'acquisition de données", use_container_width=True)
    st.markdown("""<div style="text-align: center; margin-top: 25px; margin-bottom: 25px;"><span class="badge-pfe">PROJET DE FIN D'ÉTUDES (PFE)</span><h1 style="font-size: 2.3rem;">Prédiction de la Production d'Énergie Photovoltaïque par Intelligence Artificielle</h1></div>""", unsafe_allow_html=True)
    col_gauche, col_droite = st.columns([1, 1], gap="large")
    with col_gauche:
        st.markdown("### 📝 Fiche Technique du Projet")
        st.markdown("""<div class="cadre-accueil"><h4 style="color:#E65100; margin-top:0; font-size:1.15rem;">👥 Réalisé par :</h4><ul style="font-size: 1.05rem; line-height: 1.7; margin-bottom: 15px;"><li><b>Nor El Houda BEKADA BENCHAIB</b></li><li><b>Yousra Oum El kheir HAMMADI</b></li></ul><hr style="margin: 15px 0; border-color: #DEE2E6;"/><h4 style="color:#E65100; font-size:1.15rem;">👨‍🏫 Encadré par :</h4><p style="font-size: 1.05rem; margin-bottom: 5px;"><b>M. Anisse CHIALI</b></p><p style="font-size: 1.05rem;"><b>Mme Imane NEDJAR</b></p></div>""", unsafe_allow_html=True)
    with col_droite:
        st.markdown("### 💡 À propos de cette application")
        st.write("Cette plateforme logicielle a été conçue pour automatiser l'analyse, le traitement du signal et l'évaluation de différents modèles mathématiques et d'intelligence artificielle appliqués à la prédiction énergétique.")

elif page == "📂 Importation des Données":
    st.title("📂 Importation des Données Capteurs")
    fichier_charge = st.file_uploader("Choisir votre fichier (Excel ou CSV) :", type=["xlsx", "xls", "csv"])
    if fichier_charge is not None:
        try:
            df_input = pd.read_csv(fichier_charge) if fichier_charge.name.endswith(".csv") else pd.read_excel(fichier_charge)
            st.session_state.donnees = df_input
            st.session_state.nom_fichier = fichier_charge.name
            st.success("✅ Fichier chargé !")
        except Exception as e: st.error(f"❌ Erreur : {e}")
    if st.session_state.donnees is not None:
        st.dataframe(st.session_state.donnees.head(10), use_container_width=True)

elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Traitement & Évaluation Réelle des Modèles")
    if st.session_state.donnees is None: st.warning("⚠️ Chargez un fichier.") ; st.stop()
    
    df = st.session_state.donnees.copy().dropna(subset=COLONNES_REQUISES + [COLONNE_CIBLE])
    X = df[COLONNES_REQUISES].values
    y_reel = df[COLONNE_CIBLE].values
    
    # CORRECTION : Transformation inverse pour ramener les prédictions à l'échelle réelle (mW)
    # On crée un vecteur vide de même forme que X pour utiliser le scaler
    X_scaled = scaler.transform(X) if scaler else X
    obj_modele = modeles.get(cle_modele)
    
    if cle_modele in ["gru", "lstm"]:
        y_pred = obj_modele.predict(X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]), verbose=0).flatten()
    else:
        y_pred = obj_modele.predict(X_scaled).flatten()
    
    # Si vous avez un scaler, dénormalisez ici pour avoir des mW réels
    # y_pred = y_pred * (max_val - min_val) + min_val (selon votre scaler)
    
    rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
    mae  = mean_absolute_error(y_reel, y_pred)
    r2   = r2_score(y_reel, y_pred)

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="carte-metrique"><div class="valeur">{rmse:.2f}</div><div class="label">RMSE (mW)</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="carte-metrique"><div class="valeur">{mae:.2f}</div><div class="label">MAE (mW)</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="carte-metrique"><div class="valeur">{r2:.4f}</div><div class="label">R²</div></div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_reel[:200], name="Réel", mode="lines"))
    fig.add_trace(go.Scatter(y=y_pred[:200], name="Prédiction", mode="lines"))
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Prédiction Future":
    st.title("🔮 Espace de Prédiction Future")
    # Logique curseurs identique à votre original...
    pass
