import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ───────────────────────── CONFIG ─────────────────────────
st.set_page_config(
    page_title="Prédiction PV par IA",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C"]
COLONNE_CIBLE = "Puissance_mW"
FICHIER_EXEMPLE = "Classeur1.xlsx"

MODELES_DISPONIBLES = {
    "📐 ARX": "arx",
    "🧠 PMC": "mlp",
    "🔁 GRU": "gru",
    "💾 LSTM": "lstm",
    "🔮 ANFIS": "anfis",
}

# ───────────────────────── CHARGEMENT MODELES ─────────────────────────
@st.cache_resource
def charger_ressources():
    modeles = {}

    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None

    for m in ["arx", "mlp", "anfis"]:
        path = f"model_{m}.pkl"
        if os.path.exists(path):
            modeles[m] = joblib.load(path)

    try:
        from tensorflow.keras.models import load_model

        for m in ["gru", "lstm"]:
            path = f"model_{m}.h5"
            if os.path.exists(path):
                modeles[m] = load_model(path, compile=False)
    except:
        pass

    return modeles, scaler


modeles, scaler = charger_ressources()

# ───────────────────────── STATE ─────────────────────────
if "donnees" not in st.session_state:
    st.session_state.donnees = None

# ───────────────────────── SIDEBAR ─────────────────────────
with st.sidebar:
    st.title("☀️ PV IA")

    page = st.radio(
        "Menu",
        [
            "🏠 Accueil & Présentation",
            "📂 Importation des Données",
            "📊 Évaluation & Graphiques",
            "🔮 Prédiction Future",
        ],
    )

    nom_modele = st.selectbox("Modèle", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele]

# ───────────────────────── PAGE ACCUEIL ─────────────────────────
if page == "🏠 Accueil & Présentation":
    st.title("Plateforme PV IA")
    st.write("Système de prédiction de puissance photovoltaïque")

# ───────────────────────── IMPORTATION ─────────────────────────
elif page == "📂 Importation des Données":
    file = st.file_uploader("Upload fichier", type=["xlsx", "csv"])

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.session_state.donnees = df
        st.success("Fichier chargé")

    if st.session_state.donnees is not None:
        st.dataframe(st.session_state.donnees.head())

# ───────────────────────── ÉVALUATION ─────────────────────────
elif page == "📊 Évaluation & Graphiques":

    if st.session_state.donnees is None:
        st.warning("Importer des données")
        st.stop()

    df = st.session_state.donnees.copy()
    model = modeles.get(cle_modele)

    if model is None:
        st.error("Modèle introuvable")
        st.stop()

    X = df[COLONNES_REQUISES].values.astype(float)
    X = np.nan_to_num(X)

    # reshape GRU/LSTM
    if cle_modele in ["gru", "lstm"]:
        X_in = X.reshape(X.shape[0], 1, X.shape[1])
    else:
        X_in = X

    y_pred = model.predict(X_in).flatten()
    y_pred = np.clip(y_pred, 0, None)

    # ✅ CORRECTION R²
    y_true = df[COLONNE_CIBLE].values.astype(float).reshape(-1)

    mask = ~np.isnan(y_true)
    y_true = y_true[mask]

    y_pred = np.array(y_pred).reshape(-1)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    st.subheader("📊 Métriques")
    st.write("RMSE:", rmse)
    st.write("MAE:", mae)
    st.write("R²:", r2)

    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, name="Réel"))
    fig.add_trace(go.Scatter(y=y_pred, name="Prédit"))
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────── PRÉDICTION FUTURE ─────────────────────────
elif page == "🔮 Prédiction Future":

    model = modeles.get(cle_modele)

    if model is None:
        st.error("Modèle manquant")
        st.stop()

    mode = st.radio("Mode", ["Point unique", "Simulation"])

    if mode == "Point unique":

        ldr = st.slider("LDR", 0, 4095, 1500)
        hum = st.slider("Humidité", 0.0, 100.0, 70.0)
        temp = st.slider("Température", -5.0, 55.0, 25.0)

        x = np.array([[ldr, hum, temp]])

        if cle_modele in ["gru", "lstm"]:
            x = x.reshape(1, 1, 3)

        pred = model.predict(x)[0]
        st.success(f"Puissance : {max(0, pred):.2f} mW")

    else:

        st.markdown("### Simulation sur données existantes (pas un vrai futur)")

        if st.session_state.donnees is None:
            st.warning("Importer données")
            st.stop()

        h = st.slider("Points", 5, 100, 30)

        df = st.session_state.donnees.head(h)

        X = df[COLONNES_REQUISES].values.astype(float)
        X = np.nan_to_num(X)

        if cle_modele in ["gru", "lstm"]:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        preds = model.predict(X).flatten()
        preds = np.clip(preds, 0, None)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=preds, mode="lines+markers"))
        st.plotly_chart(fig, use_container_width=True)
