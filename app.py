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

st.markdown(
    """
    <style>
    :root {
        --couleur-primaire   : #FF6B2B;
        --couleur-secondaire : #1A1A2E;
        --couleur-accent     : #FFC107;
        --couleur-fond       : #0F0F1A;
        --couleur-carte      : #16213E;
        --couleur-texte      : #E8E8F0;
        --couleur-succes     : #00C48C;
        --couleur-info       : #4FC3F7;
    }

    .stApp {
        background: linear-gradient(135deg, #0F0F1A 0%, #16213E 50%, #0F3460 100%);
        color: var(--couleur-texte);
        font-family: 'Segoe UI', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
        border-right: 2px solid var(--couleur-primaire);
    }

    h1, h2, h3 {
        color: var(--couleur-accent) !important;
    }

    .carte-metrique {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 1px solid var(--couleur-primaire);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(255, 107, 43, 0.15);
        margin-bottom: 12px;
    }
    .carte-metrique .valeur {
        font-size: 2rem;
        font-weight: 700;
        color: var(--couleur-primaire);
    }
    .carte-metrique .label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .resultat-prediction {
        background: linear-gradient(135deg, #FF6B2B22, #FFC10722);
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
    .resultat-prediction .unite {
        font-size: 1.5rem;
        color: #aaa;
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
        box-shadow: 0 6px 20px rgba(255, 107, 43, 0.4);
    }

    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px;
    }

    hr {
        border-color: var(--couleur-primaire) !important;
        opacity: 0.3;
    }

    .stDataFrame {
        border: 1px solid #333;
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
        st.warning("⚠️ Fichier 'scaler.pkl' introuvable. La normalisation sera ignorée.")

    fichiers_pkl = {
        "arx"   : "modèle_arx.pkl",
        "mlp"   : "modèle_mlp.pkl",
        "anfis" : "modèle_anfis.pkl",
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
            "gru"  : "modèle_gru.h5",
            "lstm" : "modèle_lstm.h5",
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
        st.warning("⚠️ TensorFlow n'est pas installé. Les modèles GRU et LSTM ne seront pas disponibles.")

    return modeles, scaler


def charger_donnees(source) -> pd.DataFrame | None:
    try:
        nom = getattr(source, "name", str(source))
        if nom.endswith(".csv"):
            df = pd.read_csv(source, sep=None, engine="python")
        else:
            df = pd.read_excel(source)

        # ── MAP GÉNÉRÉ DIRECTEMENT DEPUIS LES COLONNES DU FICHIER EXCEL ──
        df = df.rename(columns={
            'Temp_C': 'Température',
            'Hum_%': 'Humidité',
            'LDR_Raw': 'Éclairage',
            'Puissance_mW': 'Puissance',
            'Puissance_n': 'Puissance'
        })
        return df
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}")
        return None


def verifier_colonnes_meteo(df: pd.DataFrame) -> bool:
    manquantes = [c for c in COLONNES_METEO if c not in df.columns]
    if manquantes:
        st.error(
            f"❌ **Colonnes manquantes dans le fichier :** `{', '.join(manquantes)}`\n\n"
            f"Le fichier doit contenir les colonnes suivantes : "
            f"**{', '.join(COLONNES_METEO)}**.\n\n"
            f"Colonnes détectées dans votre fichier : `{', '.join(df.columns.tolist())}`"
        )
        return False
    return True


def normaliser_donnees(df: pd.DataFrame, scaler) -> np.ndarray | None:
    X = df[COLONNES_METEO].values.astype(float)
    if scaler is not None:
        try:
            X_norm = scaler.transform(X)
            return X_norm
        except Exception as e:
            st.warning(f"⚠️ Erreur lors de la normalisation : {e}. Les données brutes seront utilisées.")
    return X


def predire(modele, cle_modele: str, X_norm: np.ndarray) -> np.ndarray | None:
    try:
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
            line=dict(color="#4FC3F7", width=2),
            fill="tozeroy",
            fillcolor="rgba(79, 195, 247, 0.06)",
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
            font=dict(size=16, color="#FFC107"),
        ),
        xaxis=dict(title="Échantillon (indice temporel)", gridcolor="#222", color="#aaa"),
        yaxis=dict(title="Puissance (kW)", gridcolor="#222", color="#aaa"),
        plot_bgcolor="#0F0F1A",
        paper_bgcolor="#16213E",
        font=dict(color="#E8E8F0"),
        legend=dict(bgcolor="#1A1A2E", bordercolor="#FF6B2B", borderwidth=1),
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
            marker=dict(size=4, color="#FFC107"),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 43, 0.1)",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Prévision de Puissance — Modèle {nom_modele}",
            font=dict(size=16, color="#FFC107"),
        ),
        xaxis=dict(title="Horizon temporel (heures)", gridcolor="#222", color="#aaa"),
        yaxis=dict(title="Puissance estimée (kW)", gridcolor="#222", color="#aaa"),
        plot_bgcolor="#0F0F1A",
        paper_bgcolor="#16213E",
        font=dict(color="#E8E8F0"),
        legend=dict(bgcolor="#1A1A2E", bordercolor="#FF6B2B", borderwidth=1),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def afficher_carte_metrique(label: str, valeur: float, unite: str = "", precision: int = 4):
    st.markdown(
        f"""
        <div class="carte-metrique">
            <div class="valeur">{valeur:.{precision}f} <span style="font-size:1rem;color:#aaa">{unite}</span></div>
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
            <h2 style="color:#FFC107; margin:0; font-size:1.1rem;">Prédiction PV par IA</h2>
            <p style="color:#888; font-size:0.75rem; margin:4px 0 0 0;">Projet de Fin d'Études</p>
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
        <div style="color:#666; font-size:0.75rem; text-align:center; padding-top:10px;">
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
        <p style="text-align:center; color:#aaa; font-size:1rem;">
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
                        background: linear-gradient(135deg, #1A1A2E, #16213E);
                        border: 2px dashed #FF6B2B44;
                        border-radius: 12px;
                        padding: 60px 20px;
                        text-align: center;
                        color: #555;
                    ">
                        <div style="font-size:3rem;">📡</div>
                        <p style="margin-top:10px;">Schéma du dispositif de capteurs<br/>
                        <small>(Remplacez par 'schema_capteurs.png' dans votre dépôt)</small></p>
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

                > Ces variables constituent les **entrées du modèle prédictif**. La puissance est
                > la **sortie cible** utilisée lors de la phase d'évaluation.
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
            help=(
                "Le fichier doit contenir les colonnes : "
                f"{', '.join(COLONNES_METEO)} "
                f"(et idéalement '{COLONNE_PUISSANCE}' pour la page d'évaluation)."
            ),
        )

    with col_info:
        st.markdown(
            """
            <div style="
                background: #1A1A2E;
                border-left: 4px solid #FFC107;
                border-radius: 8px;
                padding: 16px;
                font-size:0.85rem;
                color:#ccc;
            ">
                <b>💡 Format requis</b><br/><br/>
                Colonnes obligatoires :<br/>
                • <code>Éclairage (LDR_Raw)</code><br/>
                • <code>Humidité (Hum_%)</code><br/>
                • <code>Température (Temp_C)</code><br/><br/>
                Colonne optionnelle (évaluation) :<br/>
                • <code>Puissance (Puissance_mW)</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if fichier_utilisateur is not None:
        df = charger_donnees(fichier_utilisateur)
        if df is not None:
            st.session_state.donnees_chargees = df
            st.session_state.nom_fichier      = fichier_utilisateur.name
            st.success(
                f"✅ **Fichier '{fichier_utilisateur.name}' chargé avec succès !** "
                f"— {len(df)} lignes × {len(df.columns)} colonnes détectées."
            )
    else:
        if st.session_state.donnees_chargees is None:
            if os.path.exists(FICHIER_EXEMPLE):
                df_exemple = charger_donnees(FICHIER_EXEMPLE)
                if df_exemple is not None:
                    st.session_state.donnees_chargees = df_exemple
                    st.session_state.nom_fichier      = FICHIER_EXEMPLE
                    st.info(
                        f"ℹ️ Aucun fichier chargé. Le fichier exemple **'{FICHIER_EXEMPLE}'** "
                        f"est utilisé par défaut ({len(df_exemple)} lignes)."
                    )
            else:
                st.warning(
                    f"⚠️ Aucun fichier chargé et le fichier exemple "
                    f"'{FICHIER_EXEMPLE}' est introuvable. Veuillez importer un fichier."
                )

    if st.session_state.donnees_chargees is not None:
        df_affiche = st.session_state.donnees_chargees
        st.divider()
        st.markdown(f"#### 🔍 Aperçu des Données — `{st.session_state.nom_fichier}`")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            afficher_carte_metrique("Nombre de lignes",    len(df_affiche),          "", 0)
        with col_s2:
            afficher_carte_metrique("Nombre de colonnes",  len(df_affiche.columns),  "", 0)
        with col_s3:
            valeurs_nulles = df_affiche.isnull().sum().sum()
            afficher_carte_metrique("Valeurs manquantes",  valeurs_nulles,            "", 0)
        with col_s4:
            has_puissance = COLONNE_PUISSANCE in df_affiche.columns
            afficher_carte_metrique(
                "Colonne Puissance",
                1 if has_puissance else 0,
                "✅" if has_puissance else "❌",
                0,
            )

        st.markdown("**Premières lignes du tableau :**")
        st.dataframe(
            df_affiche.head(20),
            use_container_width=True,
            hide_index=False,
        )

        with st.expander("📈 Statistiques Descriptives (cliquez pour déplier)"):
            st.dataframe(df_affiche.describe().round(4), use_container_width=True)

        cols_disponibles = [c for c in COLONNES_METEO if c in df_affiche.columns]
        if cols_disponibles:
            with st.expander("📊 Visualisation Rapide des Séries Temporelles"):
                for col in cols_disponibles:
                    fig_mini = px.line(
                        df_affiche,
                        y=col,
                        title=f"Évolution de : {col}",
                        color_discrete_sequence=["#FF6B2B"],
                    )
                    fig_mini.update_layout(
                        plot_bgcolor="#0F0F1A",
                        paper_bgcolor="#16213E",
                        font_color="#E8E8F0",
                        height=250,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)


elif page_choisie == "📊 Évaluation des Modèles":

    st.markdown(
        """
        <h1 style="text-align:center;">📊 Traitement & Évaluation des Modèles d'IA</h1>
        <p style="text-align:center; color:#aaa; font-size:1rem;">
            Validation croisée : puissance réelle mesurée vs puissance prédite par l'IA.
        </p>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.donnees_chargees is None:
        st.warning(
            "⚠️ **Aucune donnée chargée.** "
            "Veuillez d'abord vous rendre sur la page **📂 Importation des Données** "
            "et charger un fichier."
        )
        st.stop()

    df = st.session_state.donnees_chargees

    if not verifier_colonnes_meteo(df):
        st.stop()

    if COLONNE_PUISSANCE not in df.columns:
        st.error(
            f"❌ **La colonne '{COLONNE_PUISSANCE}' est absente** dans le fichier chargé.\n\n"
            "Cette page nécessite la puissance réelle mesurée pour effectuer la validation. "
            "Veuillez charger un fichier contenant cette colonne, ou utiliser la page "
            "**🔮 Prédiction Future** pour travailler sans données de référence."
        )
        st.stop()

    modele_actif = modeles.get(cle_modele)
    if modele_actif is None:
        st.error(
            f"❌ Le modèle **{nom_modele_court}** n'est pas disponible. "
            "Vérifiez que le fichier correspondant est présent dans votre dépôt."
        )
        st.stop()

    with st.spinner("⚙️ Normalisation des données météorologiques…"):
        X_norm = normaliser_donnees(df, scaler)

    with st.spinner(f"🤖 Calcul des prédictions avec le modèle {nom_modele_court}…"):
        y_predit = predire(modele_actif, cle_modele, X_norm)

    if y_predit is None:
        st.stop()

    y_reel = df[COLONNE_PUISSANCE].values.astype(float)

    n_min    = min(len(y_reel), len(y_predit))
    y_reel   = y_reel[:n_min]
    y_predit = y_predit[:n_min]

    metriques = calculer_metriques(y_reel, y_predit)

    st.markdown(f"### 📐 Indicateurs de Performance — {nom_modele_court}")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        afficher_carte_metrique("RMSE — Erreur Quadratique Moyenne", metriques["RMSE"], "kW")
    with col_m2:
        afficher_carte_metrique("MAE — Erreur Absolue Moyenne",      metriques["MAE"],  "kW")
    with col_m3:
        afficher_carte_metrique("R² — Coefficient de Détermination", metriques["R²"],  "",  4)

    r2 = metriques["R²"]
    if r2 >= 0.95:
        qualite = "🟢 Excellente précision"
    elif r2 >= 0.85:
        qualite = "🟡 Bonne précision"
    elif r2 >= 0.70:
        qualite = "🟠 Précision acceptable"
    else:
        qualite = "🔴 Précision insuffisante"

    st.info(f"**Qualité du modèle {nom_modele_court} :** {qualite} (R² = {r2:.4f})")

    st.divider()

    st.markdown("### 📈 Graphique de Validation")
    fig_comparaison = creer_graphique_comparaison(y_reel, y_predit, nom_modele_court)
    st.plotly_chart(fig_comparaison, use_container_width=True)

    with st.expander("🔎 Graphique de Parité (Réel vs Prédit)"):
        fig_parite = go.Figure()
        fig_parite.add_trace(
            go.Scatter(
                x=y_reel,
                y=y_predit,
                mode="markers",
                marker=dict(color="#FF6B2B", size=5, opacity=0.7),
                name="Points de mesure",
            )
        )
        axe_min = float(min(y_reel.min(), y_predit.min()))
        axe_max = float(max(y_reel.max(), y_predit.max()))
        fig_parite.add_trace(
            go.Scatter(
                x=[axe_min, axe_max],
                y=[axe_min, axe_max],
                mode="lines",
                line=dict(color="#4FC3F7", dash="dash", width=1.5),
                name="Prédiction parfaite",
            )
        )
        fig_parite.update_layout(
            title="Graphique de Parité : Puissance Réelle vs Prédite",
            xaxis_title="Puissance Réelle (kW)",
            yaxis_title="Puissance Prédite (kW)",
            plot_bgcolor="#0F0F1A",
            paper_bgcolor="#16213E",
            font_color="#E8E8F0",
        )
        st.plotly_chart(fig_parite, use_container_width=True)

    with st.expander("📋 Tableau Comparatif Numérique (30 premières lignes)"):
        df_comparatif = pd.DataFrame(
            {
                "Puissance Réelle (kW)": y_reel[:30].round(4),
                "Puissance Prédite (kW)": y_predit[:30].round(4),
                "Erreur Absolue (kW)": np.abs(y_reel[:30] - y_predit[:30]).round(4),
                "Erreur Relative (%)": (
                    np.abs(y_reel[:30] - y_predit[:30])
                    / (np.abs(y_reel[:30]) + 1e-9)
                    * 100
                ).round(2),
            }
        )
        st.dataframe(df_comparatif, use_container_width=True)

    st.divider()
    st.markdown("### 💾 Exporter les Résultats")
    df_export = pd.DataFrame(
        {
            "Puissance_Reelle_kW": y_reel,
            "Puissance_Predite_kW": y_predit,
            "Erreur_Absolue_kW": np.abs(y_reel - y_predit),
        }
    )
    buffer = io.BytesIO()
    df_export.to_excel(buffer, index=False)
    st.download_button(
        label="📥 Télécharger les résultats (Excel)",
        data=buffer.getvalue(),
        file_name=f"resultats_{cle_modele}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


elif page_choisie == "🔮 Prédiction Future":
    st.markdown(
        """
        <h1 style="text-align:center;">🔮 Prédiction Future de Puissance PV</h1>
        <p style="text-align:center; color:#aaa; font-size:1rem;">
            Estimez la production électrique à partir de nouvelles prévisions météorologiques.
        </p>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.donnees_chargees is None:
        st.warning(
            "⚠️ **Aucune donnée chargée.** "
            "Veuillez d'abord vous rendre sur la page **📂 Importation des Données**."
        )
        st.stop()

    df = st.session_state.donnees_chargees

    if not verifier_colonnes_meteo(df):
        st.stop()

    modele_actif = modeles.get(cle_modele)
    if modele_actif is None:
        st.error(f"❌ Le modèle **{nom_modele_court}** n'est pas disponible.")
        st.stop()

    with st.spinner("⚙️ Traitement des données…"):
        X_norm = normaliser_donnees(df, scaler)
        y_predit = predire(modele_actif, cle_modele, X_norm)

    if y_predit is not None:
        st.markdown(f"### 🔮 Horizons de Production Estimés — {nom_modele_court}")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            afficher_carte_metrique("Production Moyenne Estimée", float(np.mean(y_predit)), "kW")
        with col_p2:
            afficher_carte_metrique("Pic de Production Maximal", float(np.max(y_predit)), "kW")

        fig_future = creer_graphique_prediction(y_predit, nom_modele_court)
        st.plotly_chart(fig_future, use_container_width=True)
