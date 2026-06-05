"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PLATEFORME DE PRÉDICTION D'ÉNERGIE PHOTOVOLTAÏQUE PAR IA            ║
║                Projet de Fin d'Études — Ingénierie des Systèmes              ║
╚══════════════════════════════════════════════════════════════════════════════╝
Fichier  : app.py
Auteur   : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI
Description :
    Application web Streamlit pour la prédiction de la puissance générée par
    un système photovoltaïque à l'aide de cinq modèles d'IA / traitement du signal :
    ARX, PMC (MLP), GRU, LSTM et ANFIS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import warnings

warnings.filterwarnings("ignore")

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Prédiction PV par IA",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN ET THÈME CLAIR (FOND BLANC, TEXTE SOMBRE, ACCENTS ORANGES) ──
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
        --couleur-succes     : #2B8A3E;
        --couleur-info       : #1A73E8;
    }

    .stApp {
        background-color: var(--couleur-fond);
        color: var(--couleur-texte);
        font-family: 'Segoe UI', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: var(--couleur-secondaire);
        border-right: 1px solid #DEE2E6;
    }

    h1, h2, h3 {
        color: var(--couleur-accent) !important;
        font-weight: 700 !important;
    }

    .carte-metrique {
        background-color: var(--couleur-carte);
        border: 1px solid #CED4DA;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    .carte-metrique .valeur {
        font-size: 2rem;
        font-weight: 700;
        color: var(--couleur-primaire);
    }
    .carte-metrique .label {
        font-size: 0.85rem;
        color: #495057;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .resultat-prediction {
        background-color: #FFF3CD;
        border: 2px solid var(--couleur-primaire);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .resultat-prediction .puissance {
        font-size: 3.5rem;
        font-weight: 800;
        color: var(--couleur-accent);
    }

    .stButton > button {
        background: linear-gradient(90deg, var(--couleur-primaire), #FF8C42);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 107, 43, 0.3);
    }

    hr {
        border-color: #DEE2E6 !important;
    }

    .stDataFrame {
        border: 1px solid #DEE2E6;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLONNES_METEO    = ["Éclairage", "Humidité", "Température"]
COLONNE_PUISSANCE = "Puissance"

FICHIER_EXEMPLE = "Classeur1.xlsx"

MODELES_DISPONIBLES = {
    "📐 ARX  — Modèle Autorégressif Linéaire" : "arx",
    "🧠 PMC  — Perceptron Multicouche"        : "mlp",
    "🔁 GRU  — Réseau de Neurones Récurrent"  : "gru",
    "💾 LSTM — Mémoire à Long Terme"          : "lstm",
    "🔮 ANFIS — Système Neuro-Flou Adaptatif" : "anfis",
}

@st.cache_resource(show_spinner="⚙️ Chargement des modèles d'IA…")
def charger_modeles():
    modeles = {}
    scaler  = None

    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
        except Exception as e:
            st.warning(f"⚠️ Impossible de charger le normalisateur : {e}")
    else:
        st.warning("⚠️ Fichier 'scaler.pkl' introuvable.")

    fichiers_pkl = {
        "arx"   : "model_arx.pkl",
        "mlp"   : "model_mlp.pkl",
        "anfis" : "model_anfis.pkl",
    }
    for cle, chemin in fichiers_pkl.items():
        if os.path.exists(chemin):
            try:
                modeles[cle] = joblib.load(chemin)
            except Exception as e:
                st.warning(f"⚠️ Erreur lors du chargement de '{chemin}' : {e}")
        else:
            st.warning(f"⚠️ Fichier '{chemin}' introuvable.")

    try:
        from tensorflow.keras.models import load_model
        fichiers_h5 = {
            "gru"  : "model_gru.h5",
            "lstm" : "model_lstm.h5",
        }
        for cle, chemin in fichiers_h5.items():
            if os.path.exists(chemin):
                try:
                    modeles[cle] = load_model(chemin, compile=False)
                except Exception as e:
                    st.warning(f"⚠️ Erreur lors du chargement de '{chemin}' : {e}")
            else:
                st.warning(f"⚠️ Fichier '{chemin}' introuvable.")
    except ImportError:
        st.warning("⚠️ TensorFlow n'est pas installé.")

    return modeles, scaler


def charger_donnees(source) -> pd.DataFrame | None:
    try:
        nom = getattr(source, "name", str(source))
        if nom.endswith(".csv"):
            df = pd.read_csv(source, sep=None, engine="python")
        else:
            df = pd.read_excel(source)

        df = df.rename(columns={
            'Temp_C': 'Température',
            'Hum_%': 'Humidité',
            'LDR_Raw': 'Éclairage',
            'Puissance_mW': 'Puissance',
            'Puissance_n': 'Puissance'
        })
        
        df = df.dropna(subset=[col for col in COLONNES_METEO if col in df.columns])
        return df
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}")
        return None


def verifier_colonnes_meteo(df: pd.DataFrame) -> bool:
    manquantes = [c for c in COLONNES_METEO if c not in df.columns]
    if manquantes:
        st.error(
            f"❌ **Colonnes manquantes dans le fichier :** `{', '.join(manquantes)}`"
        )
        return False
    return True


def normaliser_donnees(df: pd.DataFrame, scaler) -> np.ndarray | None:
    X = df[COLONNES_METEO].values.astype(float)
    if scaler is not None:
        try:
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != X.shape[1]:
                return X
            X_norm = scaler.transform(X)
            return X_norm
        except Exception as e:
            pass
    return X


def predire(modele, cle_modele: str, X_norm: np.ndarray) -> np.ndarray | None:
    try:
        # === CORRECTION DES NaN ET SUPPRESSION DES LIGNES COMPROMISES ===
        # Si la matrice contient des valeurs manquantes (NaN ou infini)
        if np.any(np.isnan(X_norm)) or np.any(np.isinf(X_norm)):
            # On remplace proprement les NaN par des zéros ou la moyenne pour que la régression ne plante pas
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Vérification des dimensions requises par le modèle Scikit-Learn (ARX / MLP)
        if hasattr(modele, "n_features_in_") and X_norm.shape[1] != modele.n_features_in_:
            # Si le modèle ARX attend 1 seule variable (ex: uniquement la puissance retardée), on ne lui passe que la 1ère colonne
            X_norm = X_norm[:, :modele.n_features_in_]

        if cle_modele in ("gru", "lstm"):
            X_3d = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
            predictions = modele.predict(X_3d, verbose=0).flatten()
        else:
            predictions = modele.predict(X_norm).flatten()
        return predictions
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul avec le modèle sélectionné : {e}")
        return None


def calculer_metriques(reelles: np.ndarray, predites: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(reelles, predites))
    mae  = mean_absolute_error(reelles, predites)
    r2   = r2_score(reelles, predites)
    return {"RMSE": rmse, "MAE": mae, "R²": r2}


def creer_graphique_comparaison(reelles: np.ndarray, predites: np.ndarray, nom_modele: str) -> go.Figure:
    indices = list(range(len(reelles)))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=indices,
            y=reelles,
            name="⚡ Puissance Réelle",
            mode="lines",
            line=dict(color="#1A73E8", width=2),
            fill="tozeroy",
            fillcolor="rgba(26, 115, 232, 0.05)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=indices,
            y=predites,
            name=f"🤖 Puissance Prédite ({nom_modele})",
            mode="lines",
            line=dict(color="#FF6B2B", width=2, dash="dot"),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Comparaison : Puissance Réelle vs Prédite — Modèle {nom_modele}",
            font=dict(size=16, color="#E65100"),
        ),
        xaxis=dict(title="Échantillon (indice temporel)", gridcolor="#E5E5E5", color="#333"),
        yaxis=dict(title="Puissance (kW)", gridcolor="#E5E5E5", color="#333"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F8F9FA",
        font=dict(color="#212529"),
        legend=dict(bgcolor="#FFFFFF", bordercolor="#DEE2E6", borderwidth=1),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def creer_graphique_prediction(predites: np.ndarray, nom_modele: str) -> go.Figure:
    indices = list(range(len(predites)))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=indices,
            y=predites,
            name=f"🔮 Puissance Prédite ({nom_modele})",
            mode="lines+markers",
            line=dict(color="#FF6B2B", width=2.5),
            marker=dict(size=4, color="#E65100"),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 43, 0.05)",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Prévision de Puissance — Modèle {nom_modele}",
            font=dict(size=16, color="#E65100"),
        ),
        xaxis=dict(title="Horizon temporel (heures)", gridcolor="#E5E
