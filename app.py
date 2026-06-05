import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

import plotly.graph_objects as go

st.set_page_config(
    page_title="Prédiction PV par IA",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>/* CSS inchangé */</style>""", unsafe_allow_html=True)

COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C"]
COLONNE_CIBLE = "Puissance_mW"
FICHIER_EXEMPLE = "Classeur1.xlsx"

MODELES_DISPONIBLES = {
    "📐 ARX  — Modèle Autorégressif Linéaire": "arx",
    "🧠 PMC  — Perceptron Multicouche": "mlp",
    "🔁 GRU  — Réseau de Neurones Récurrent": "gru",
    "💾 LSTM — Mémoire à Long Terme": "lstm",
    "🔮 ANFIS — Système Neuro-Flou Adaptatif": "anfis",
}

@st.cache_resource(show_spinner="⚙️ Chargement des architectures d'IA…")
def charger_ressources():
    modeles = {}
    scaler = None

    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")

    fichiers = {
        "arx": "model_arx.pkl",
        "mlp": "model_mlp.pkl",
        "anfis": "model_anfis.pkl",
    }

    for cle, chemin in fichiers.items():
        if os.path.exists(chemin):
            modeles[cle] = joblib.load(chemin)

    try:
        from tensorflow.keras.models import load_model

        modeles_h5 = {
            "gru": "model_gru.h5",
            "lstm": "model_lstm.h5",
        }

        for cle, chemin in modeles_h5.items():
            if os.path.exists(chemin):
                modeles[cle] = load_model(chemin, compile=False)
    except:
        pass

    return modeles, scaler


def preparer_matrice_entrees(df_data, scaler_obj, cle_mod, nb_attendues):
    X_base = df_data[COLONNES_REQUISES].values.astype(float)

    if np.any(np.isnan(X_base)):
        X_base = np.nan_to_num(X_base, nan=0.0)

    if scaler_obj is not None:
        try:
            if hasattr(scaler_obj, "n_features_in_"):
                if scaler_obj.n_features_in_ == X_base.shape[1]:
                    X_base = scaler_obj.transform(X_base)
        except:
            pass

    n_echantillons, n_feats = X_base.shape

    if n_feats == nb_attendues:
        return X_base
    elif nb_attendues > n_feats:
        X_adapte = np.zeros((n_echantillons, nb_attendues))
        X_adapte[:, :n_feats] = X_base
        return X_adapte
    else:
        return X_base[:, :nb_attendues]


modeles, scaler = charger_ressources()

if "donnees" not in st.session_state:
    st.session_state.donnees = None

with st.sidebar:
    page = st.radio(
        "Menu",
        [
            "🏠 Accueil & Présentation",
            "📂 Importation des Données",
            "📊 Évaluation & Graphiques",
            "🔮 Prédiction Future",
        ],
    )

    nom_modele_selectionne = st.selectbox(
        "Modèle :", list(MODELES_DISPONIBLES.keys())
    )
    cle_modele = MODELES_DISPONIBLES[nom_modele_selectionne]
    nom_court = nom_modele_selectionne.split("—")[0].strip()


# ───────────────────────── PAGE ÉVALUATION ─────────────────────────
elif page == "📊 Évaluation & Graphiques":

    df = st.session_state.donnees.copy()
    obj_modele = modeles.get(cle_modele)

    if obj_modele is None:
        st.stop()

    n_attendues = (
        obj_modele.n_features_in_
        if hasattr(obj_modele, "n_features_in_")
        else 3
    )

    X_final = preparer_matrice_entrees(df, scaler, cle_modele, n_attendues)

    if cle_modele in ["gru", "lstm"]:
        X_final = X_final.reshape(X_final.shape[0], 1, X_final.shape[1])

    y_pred = obj_modele.predict(X_final).flatten()
    y_pred = np.clip(y_pred, 0, None)

    # ---------------- CORRECTION ICI ----------------
    y_reel = df[COLONNE_CIBLE].values.astype(float).reshape(-1)

    mask = ~np.isnan(y_reel)
    y_reel = y_reel[mask]

    y_pred = np.array(y_pred).reshape(-1)

    taille_min = min(len(y_reel), len(y_pred))
    y_reel = y_reel[:taille_min]
    y_pred = y_pred[:taille_min]
    # -------------------------------------------------

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
    mae = mean_absolute_error(y_reel, y_pred)
    r2 = r2_score(y_reel, y_pred)

    st.write(rmse, mae, r2)


# ───────────────────────── PAGE FUTURE ─────────────────────────
elif page == "🔮 Prédiction Future":

    obj_modele = modeles.get(cle_modele)

    n_attendues = (
        obj_modele.n_features_in_
        if hasattr(obj_modele, "n_features_in_")
        else 3
    )

    mode_test = st.radio(
        "Méthode :",
        [
            "Saisie manuelle",
            "Simulation sur horizon temporel futur",
        ],
    )

    if mode_test == "Saisie manuelle":

        val_ldr = st.slider("LDR", 0, 4095, 1400)
        val_hum = st.slider("Hum", 0.0, 100.0, 70.0)
        val_temp = st.slider("Temp", -5.0, 55.0, 25.0)

        df_temp = pd.DataFrame(
            [{"LDR_Raw": val_ldr, "Hum_%": val_hum, "Temp_C": val_temp}]
        )

        X_pt = preparer_matrice_entrees(df_temp, scaler, cle_modele, n_attendues)

        if cle_modele in ["gru", "lstm"]:
            X_pt = X_pt.reshape(1, 1, X_pt.shape[1])

        pred = obj_modele.predict(X_pt)[0]
        st.success(f"{max(0, pred):.2f} mW")

    else:

        # ---------------- CORRECTION TEXTE ICI ----------------
        st.markdown("### Simulation (projection sur données existantes)")

        horizon = st.slider("Nombre de points :", 5, 100, 30)

        df_horizon = st.session_state.donnees.copy().head(horizon)

        X_hor = preparer_matrice_entrees(
            df_horizon, scaler, cle_modele, n_attendues
        )

        if cle_modele in ["gru", "lstm"]:
            X_hor = X_hor.reshape(X_hor.shape[0], 1, X_hor.shape[1])

        preds = obj_modele.predict(X_hor).flatten()
        preds = np.clip(preds, 0, None)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=preds, mode="lines+markers", name="Prédiction")
        )
        st.plotly_chart(fig, use_container_width=True)
