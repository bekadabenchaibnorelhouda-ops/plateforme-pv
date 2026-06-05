"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PLATEFORME DE PRÉDICTION D'ÉNERGIE PHOTOVOLTAÏQUE PAR IA            ║
║                Projet de Fin d'Études — Ingénierie des Systèmes              ║
╚══════════════════════════════════════════════════════════════════════════════╝
Fichier  : app.py
Auteur   : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI
Description :
    Application web Streamlit pour la prédiction de la puissance générée par
    un système photovoltaïque à l'aide de cinq modèles d'IA / traitement du signal.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Prédiction PV par IA",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN ET THÈME CLAIR ──
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
    .stButton > button {
        background: linear-gradient(90deg, var(--couleur-primaire), #FF8C42); color: white;
        border: none; border-radius: 8px; font-weight: 600; padding: 10px 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Correspondance exacte des colonnes détectées dans votre Excel
COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C"]
COLONNE_CIBLE     = "Puissance_mW"
FICHIER_EXEMPLE   = "Classeur1.xlsx"

MODELES_DISPONIBLES = {
    "📐 ARX  — Modèle Autorégressif Linéaire" : "arx",
    "🧠 PMC  — Perceptron Multicouche"        : "mlp",
    "🔁 GRU  — Réseau de Neurones Récurrent"  : "gru",
    "💾 LSTM — Mémoire à Long Terme"          : "lstm",
    "🔮 ANFIS — Système Neuro-Flou Adaptatif" : "anfis",
}

@st.cache_resource(show_spinner="⚙️ Chargement des architectures d'IA…")
def charger_ressources():
    """Charge le scaler et les modèles avec les noms exacts présents sur votre GitHub."""
    modeles = {}
    scaler = None

    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
        except Exception as e:
            st.warning(f"ℹ️ Erreur lecture scaler.pkl : {e}")

    # Dictionnaire calé sur vos fichiers réels (avec accents)
    fichiers = {
        "arx": "modèle_arx.pkl",
        "mlp": "modèle_mlp.pkl",
        "anfis": "modèle_anfis.pkl",
    }
    
    for cle, chemin in fichiers.items():
        if os.path.exists(chemin):
            try:
                modeles[cle] = joblib.load(chemin)
            except Exception as e:
                st.error(f"❌ Erreur sur {chemin} : {e}")
        else:
            st.sidebar.warning(f"⚠️ {chemin} absent.")

    # Chargement Deep Learning (Keras)
    try:
        from tensorflow.keras.models import load_model
        fichiers_h5 = {
            "gru": "modèle_gru.h5",
            "lstm": "modèle_lstm.h5",
        }
        for cle, chemin in fichiers_h5.items():
            if os.path.exists(chemin):
                try:
                    modeles[cle] = load_model(chemin, compile=False)
                except Exception as e:
                    st.error(f"❌ Erreur sur {chemin} : {e}")
            else:
                st.sidebar.warning(f"⚠️ {chemin} absent.")
    except ImportError:
        pass

    return modeles, scaler


def preparer_matrice_entrees(df: pd.DataFrame, scaler, cle_modele, nb_features_attendues) -> np.ndarray:
    """
    Prépare et adapte dynamiquement la matrice d'entrée X pour correspondre 
    exactement au nombre de features attendu par le modèle chargé.
    """
    # 1. Extraction des variables de base
    X_base = df[COLONNES_REQUISES].values.astype(float)
    
    # Remplacement préventif des NaN
    if np.any(np.isnan(X_base)):
        X_base = np.nan_to_num(X_base, nan=0.0)

    # 2. Application du scaler si disponible
    if scaler is not None:
        try:
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == X_base.shape[1]:
                X_base = scaler.transform(X_base)
            elif hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == 1:
                # Si le scaler n'a été entraîné que sur la première feature (LDR_Raw)
                X_base[:, 0] = scaler.transform(X_base[:, :1]).flatten()
        except Exception:
            pass

    # 3. Alignement structurel avec la dimension du modèle (ex: 5 features pour votre ARX)
    n_echantillons, n_feats = X_base.shape
    
    if n_feats == nb_features_attendues:
        return X_base
    
    elif nb_features_attendues > n_feats:
        # Cas typique de votre modèle ARX : il attend des retards (lags) temporels ou des colonnes additionnelles.
        # On complète les colonnes manquantes avec des décalages (lags) de la première colonne (LDR_Raw)
        X_adapte = np.zeros((n_echantillons, nb_features_attendues))
        X_adapte[:, :n_feats] = X_base
        
        # Remplissage des colonnes suivantes par les valeurs décalées (historique)
        for i in range(n_feats, nb_features_attendues):
            decalage = i - n_feats + 1
            X_adapte[decalage:, i] = X_base[:-decalage, 0]
            X_adapte[:decalage, i] = X_base[0, 0] # Complétion des premiers indices
        return X_adapte
        
    else:
        # Si le modèle attend moins de features, on tronque
        return X_base[:, :nb_features_attendues]


# Chargement global des fichiers
modeles, scaler = charger_ressources()

# Gestion des sessions Streamlit pour stocker le fichier
if "donnees" not in st.session_state:
    st.session_state.donnees = None
if "nom_fichier" not in st.session_state:
    st.session_state.nom_fichier = None

# ── NAVIGATION DE LA BARRE LATÉRALE ──
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>☀️ menu</h2>", unsafe_allow_html=True)
    page = st.radio("Aller à :", ["📂 Importation des Données", "📊 Évaluation & Graphiques"])
    st.divider()
    
    nom_modele_selectionne = st.selectbox("Sélectionner le modèle d'IA :", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele_selectionne]
    nom_court = nom_modele_selectionne.split("—")[0].strip()


# ── PAGE 1 : IMPORTATION ──
if page == "📂 Importation des Données":
    st.title("📂 Importation des Données Capteurs")
    
    fichier_charge = st.file_uploader("Importer votre fichier Excel ou CSV :", type=["xlsx", "xls", "csv"])
    
    if fichier_charge is not None:
        try:
            if fichier_charge.name.endswith(".csv"):
                df_input = pd.read_csv(fichier_charge, sep=None, engine="python")
            else:
                df_input = pd.read_excel(fichier_charge)
                
            st.session_state.donnees = df_input
            st.session_state.nom_fichier = fichier_charge.name
            st.success(f"✅ Fichier '{fichier_charge.name}' chargé avec succès !")
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier : {e}")
            
    elif st.session_state.donnees is None and os.path.exists(FICHIER_EXEMPLE):
        try:
            st.session_state.donnees = pd.read_excel(FICHIER_EXEMPLE)
            st.session_state.nom_fichier = FICHIER_EXEMPLE
            st.info(f"ℹ️ Mode démo : Utilisation du fichier '{FICHIER_EXEMPLE}'")
        except Exception:
            pass

    # Visualisation des données brutes importées
    if st.session_state.donnees is not None:
        st.markdown(f"### 📋 Aperçu du fichier : `{st.session_state.nom_fichier}`")
        st.dataframe(st.session_state.donnees.head(
