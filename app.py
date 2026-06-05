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
                # Si le scaler attend 1 variable mais qu'on en donne 3, on n'utilise que la première colonne (l'Éclairage)
                return scaler.transform(X[:, :scaler.n_features_in_])
            X_norm = scaler.transform(X)
            return X_norm
        except Exception as e:
            pass
    return X


def predire(modele, cle_modele: str, X_norm: np.ndarray) -> np.ndarray | None:
    try:
        # === NETTOYAGE SÉCURISÉ DES VALEURS MANQUANTES (NaN / INF) ===
        if np.any(np.isnan(X_norm)) or np.any(np.isinf(X_norm)):
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # === AJUSTEMENT DYNAMIQUE DE LA DIMENSION DES ENTRÉES ===
        # Si le modèle possède l'attribut n_features_in_ (modèles Scikit-Learn comme ARX ou MLP)
        if hasattr(modele, "n_features_in_"):
            attendu = modele.n_features_in_
            if X_norm.shape[1] != attendu:
                # Si le modèle n'attend qu'une variable (ex: ARX avec 1 seule feature), on ne garde que la première colonne
                X_norm = X_norm[:, :attendu]
        
        # Si c'est un modèle linéaire de type LinearRegression classique sans n_features_in_
        elif hasattr(modele, "coef_"):
            attendu = len(modele.coef_) if np.ndim(modele.coef_) == 1 else modele.coef_.shape[1]
            if X_norm.shape[1] != attendu:
                X_norm = X_norm[:, :attendu]

        # === EXÉCUTION DES PRÉDICTIONS ===
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
        xaxis=dict(title="Horizon temporel (échantillons)", gridcolor="#E5E5E5", color="#333"),
        yaxis=dict(title="Puissance Prédite (kW)", gridcolor="#E5E5E5", color="#333"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F8F9FA",
        font=dict(color="#212529"),
        legend=dict(bgcolor="#FFFFFF", bordercolor="#DEE2E6", borderwidth=1),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def afficher_carte_metrique(label: str, valeur: float, unite: str = "", precision: int = 4):
    st.markdown(
        f"""
        <div class="carte-metrique">
            <div class="valeur">{valeur:.{precision}f} <span style="font-size:1rem;color:#495057">{unite}</span></div>
            <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <div style="font-size:3rem;">☀️</div>
            <h2 style="color:#E65100; margin:0; font-size:1.1rem;">Prédiction PV par IA</h2>
            <p style="color:#666; font-size:0.75rem; margin:4px 0 0 0;">Projet de Fin d'Études</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 🗂️ Navigation")
    page_choisie = st.radio(
        label="Choisir une étape :",
        options=[
            "📂 Importation des Données",
            "📊 Évaluation des Modèles",
            "🔮 Prédiction Future",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### 🤖 Modèle d'IA Actif")
    nom_modele_affiche = st.selectbox(
        label="Choisir un modèle :",
        options=list(MODELES_DISPONIBLES.keys()),
        label_visibility="collapsed",
    )
    cle_modele = MODELES_DISPONIBLES[nom_modele_affiche]
    nom_modele_court = nom_modele_affiche.split("—")[0].strip()

    st.divider()

    st.markdown(
        """
        <div style="color:#495057; font-size:0.75rem; text-align:center; padding-top:10px;">
            <p>🔬 Variables d'entrée :</p>
            <p>🌡️ Température (°C)</p>
            <p>💧 Humidité (%)</p>
            <p>☀️ Éclairage (lux)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

modeles, scaler = charger_modeles()

if "donnees_chargees" not in st.session_state:
    st.session_state.donnees_chargees = None
if "nom_fichier" not in st.session_state:
    st.session_state.nom_fichier = None


if page_choisie == "📂 Importation des Données":
    st.markdown(
        """
        <h1 style="text-align:center;">📂 Importation & Configuration des Données</h1>
        <p style="text-align:center; color:#495057; font-size:1rem;">
            Chargez votre propre jeu de données ou explorez notre exemple interactif.
        </p>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        col_img, col_desc = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("#### 🖼️ Protocole de Collecte des Données")
            if os.path.exists("schema_capteurs.png"):
                st.image(
                    "schema_capteurs.png",
                    caption="Schéma du dispositif de mesure météorologique",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    """
                    <div style="
                        background-color: #F1F3F5;
                        border: 2px dashed #CED4DA;
                        border-radius: 12px;
                        padding: 60px 20px;
                        text-align: center;
                        color: #6C757D;
                    ">
                        <div style="font-size:3rem;">📡</div>
                        <p style="margin-top:10px;"><b>Schéma introuvable</b><br/>
                        Placez l'image sous le nom <code>schema_capteurs.png</code> dans votre dossier.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_desc:
            st.markdown("#### 📋 Variables Mesurées")
            st.markdown(
                """
                Les trois grandeurs physiques acquises par le dispositif expérimental sont :

                | Variable | Capteur | Unité |
                |---|---|---|
                | 🌡️ **Température** | Sonde PT100 / DHT22 | °C |
                | 💧 **Humidité relative** | Capteur DHT22 | % |
                | ☀️ **Éclairage (irradiance)** | Capteur LDR / pyranomètre | lux |
                | ⚡ **Puissance générée** | Wattmètre numérique | kW |
                """
            )
            if os.path.exists("installation_pv.jpg"):
                st.image(
                    "installation_pv.jpg",
                    caption="Installation photovoltaïque instrumentée",
                    use_container_width=True,
                )

    st.divider()
    st.markdown("#### 📤 Charger votre Fichier de Données")
    
    col_upload, col_info = st.columns([2, 1], gap="large")
    with col_upload:
        fichier_utilisateur = st.file_uploader(
            label="Sélectionner un fichier Excel (.xlsx) ou CSV (.csv) :",
            type=["xlsx", "xls", "csv"],
        )

    with col_info:
        st.markdown(
            """
            <div style="background-color: #F8F9FA; border-left: 4px solid #FF6B2B; padding: 16px; font-size:0.85rem; color:#212529;">
                <b>💡 Format requis</b><br/>
                • Éclairage (LDR_Raw)<br/>
                • Humidité (Hum_%)<br/>
                • Température (Temp_C)
            </div>
            """,
            unsafe_allow_html=True,
        )

    if fichier_utilisateur is not None:
        df = charger_donnees(fichier_utilisateur)
        if df is not None:
            st.session_state.donnees_chargees = df
            st.session_state.nom_fichier      = fichier_utilisateur.name
            st.success(f"✅ **Fichier '{fichier_utilisateur.name}' chargé !**")
    else:
        if st.session_state.donnees_chargees is None and os.path.exists(FICHIER_EXEMPLE):
            df_exemple = charger_donnees(FICHIER_EXEMPLE)
            if df_exemple is not None:
                st.session_state.donnees_chargees = df_exemple
                st.session_state.nom_fichier      = FICHIER_EXEMPLE
                st.info(f"ℹ️ Utilisation du fichier exemple : **'{FICHIER_EXEMPLE}'**")

    if st.session_state.donnees_chargees is not None:
        df_affiche = st.session_state.donnees_chargees
        st.divider()
        st.markdown(f"#### 🔍 Aperçu des Données — `{st.session_state.nom_fichier}`")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            afficher_carte_metrique("Nombre de lignes", len(df_affiche), "", 0)
        with col_s2:
            afficher_carte_metrique("Nombre de colonnes", len(df_affiche.columns), "", 0)
        with col_s3:
            afficher_carte_metrique("Valeurs manquantes", df_affiche.isnull().sum().sum(), "", 0)
        with col_s4:
            has_puissance = COLONNE_PUISSANCE in df_affiche.columns
            afficher_carte_metrique("Colonne Puissance", 1 if has_puissance else 0, "✅" if has_puissance else "❌", 0)

        st.dataframe(df_affiche.head(20), use_container_width=True)


elif page_choisie == "📊 Évaluation des Modèles":
    st.markdown(
        """
        <h1 style="text-align:center;">📊 Traitement & Évaluation des Modèles d'IA</h1>
        <p style="text-align:center; color:#495057; font-size:1rem;">
            Validation croisée : puissance réelle mesurée vs puissance prédite par l'IA.
        </p>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.donnees_chargees is None:
        st.warning("⚠️ **Aucune donnée chargée.** Veuillez d'abord charger un fichier en page 1.")
        st.stop()

    df = st.session_state.donnees_chargees

    if not verifier_colonnes_meteo(df):
        st.stop()

    if COLONNE_PUISSANCE not in df.columns:
        st.error(f"❌ **La colonne '{COLONNE_PUISSANCE}' est absente** pour l'évaluation.")
        st.stop()

    modele_actif = modeles.get(cle_modele)
    if modele_actif is None:
        st.error(f"❌ Le modèle **{nom_modele_court}** n'est pas disponible (Vérifiez les fichiers 'model_*.pkl/h5').")
        st.stop()

    with st.spinner("⚙️ Normalisation des données…"):
        X_norm = normaliser_donnees(df, scaler)

    with st.spinner(f"🤖 Calcul des prédictions…"):
        y_predit = predire(modele_actif, cle_modele, X_norm)

    if y_predit is None:
        st.stop()

    y_reel = df[COLONNE_PUISSANCE].values.astype(float)
    n_min = min(len(y_reel), len(y_predit))
    y_reel, y_predit = y_reel[:n_min], y_predit[:n_min]

    metriques = calculer_metriques(y_reel, y_predit)

    st.markdown(f"### 📐 Indicateurs de Performance — {nom_modele_court}")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        afficher_carte_metrique("RMSE (Erreur Quadratique Moyenne)", metriques["RMSE"], "kW")
    with col_m2:
        afficher_carte_metrique("MAE (Erreur Absolue Moyenne)",      metriques["MAE"],  "kW")
    with col_m3:
        afficher_carte_metrique("R² (Coefficient de Détermination)", metriques["R²"],  "",  4)

    st.divider()
    st.markdown("### 📈 Graphique de Validation")
    fig_comparaison = creer_graphique_comparaison(y_reel, y_predit, nom_modele_court)
    st.plotly_chart(fig_comparaison, use_container_width=True)

    with st.expander("📋 Tableau Comparatif Numérique"):
        df_comparatif = pd.DataFrame({
            "Puissance Réelle (kW)": y_reel[:30].round(4),
            "Puissance Prédite (kW)": y_predit[:30].round(4),
            "Erreur Absolue (kW)": np.abs(y_reel[:30] - y_predit[:30]).round(4),
        })
        st.dataframe(df_comparatif, use_container_width=True)


elif page_choisie == "🔮 Prédiction Future":
    st.markdown(
        """
        <h1 style="text-align:center;">🔮 Prédiction Future de Puissance PV</h1>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.donnees_chargees is None:
        st.warning("⚠️ **Aucune donnée chargée.**")
        st.stop()

    df = st.session_state.donnees_chargees

    if not verifier_colonnes_meteo(df):
        st.stop()

    modele_actif = modeles.get(cle_modele)
    if modele_actif is None:
        st.error(f"❌ Modèle introuvable.")
        st.stop()

    X_norm = normaliser_donnees(df, scaler)
    y_predit = predire(modele_actif, cle_modele, X_norm)

    if y_predit is not None:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            afficher_carte_metrique("Production Moyenne Estimée", float(np.mean(y_predit)), "kW")
        with col_p2:
            afficher_carte_metrique("Pic de Production Maximal", float(np.max(y_predit)), "kW")

        fig_future = creer_graphique_prediction(y_predit, nom_modele_court)
        st.plotly_chart(fig_future, use_container_width=True)
