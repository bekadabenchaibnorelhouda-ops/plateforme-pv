"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PLATEFORME DE PRÉDICTION D'ÉNERGIE PHOTOVOLTAÏQUE PAR IA            ║
║                  Projet de Fin d'Études — Ingénierie des Systèmes           ║
╚══════════════════════════════════════════════════════════════════════════════╝
Fichier  : app.py
Auteur   : [Votre Nom]
Date     : 2025
Description :
    Application web Streamlit pour la prédiction de la puissance générée par
    un système photovoltaïque à l'aide de cinq modèles d'IA/traitement du signal :
    ARX, PMC (MLP), GRU, LSTM et ANFIS.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS ET CONFIGURATION GLOBALE
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import warnings

warnings.filterwarnings("ignore")

# Visualisation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Métriques
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Configuration de la page Streamlit ──────────────────────────────────────
st.set_page_config(
    page_title="Prédiction PV par IA",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Injection de CSS personnalisé ────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Palette de couleurs ── */
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

    /* ── Fond général ── */
    .stApp {
        background: linear-gradient(135deg, #0F0F1A 0%, #16213E 50%, #0F3460 100%);
        color: var(--couleur-texte);
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Barre latérale ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
        border-right: 2px solid var(--couleur-primaire);
    }

    /* ── Titres principaux ── */
    h1, h2, h3 {
        color: var(--couleur-accent) !important;
    }

    /* ── Cartes de métriques ── */
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

    /* ── Conteneur résultat prédiction ── */
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

    /* ── Boutons ── */
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

    /* ── Messages d'alerte et succès ── */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px;
    }

    /* ── Séparateur ── */
    hr {
        border-color: var(--couleur-primaire) !important;
        opacity: 0.3;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border: 1px solid #333;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTES ET CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Noms des colonnes attendues dans les fichiers de données
COLONNES_METEO    = ["Éclairage", "Humidité", "Température"]
COLONNE_PUISSANCE = "Puissance"

# Chemin du fichier de démonstration par défaut
FICHIER_EXEMPLE = "Classeur1.xlsx"

# Dictionnaire des modèles disponibles
MODELES_DISPONIBLES = {
    "📐 ARX  — Modèle Autorégressif Linéaire" : "arx",
    "🧠 PMC  — Perceptron Multicouche"        : "mlp",
    "🔁 GRU  — Réseau de Neurones Récurrent"  : "gru",
    "💾 LSTM — Mémoire à Long Terme"          : "lstm",
    "🔮 ANFIS — Système Neuro-Flou Adaptatif" : "anfis",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. FONCTIONS UTILITAIRES
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ Chargement des modèles d'IA…")
def charger_modeles():
    """
    Charge en mémoire le normalisateur (scaler) et les cinq modèles entraînés.
    Utilise @st.cache_resource pour n'effectuer le chargement qu'une seule fois.
    Retourne un dictionnaire {nom_court: modèle} et le scaler.
    """
    modeles = {}
    scaler  = None

    # ── Chargement du normalisateur ──
    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
        except Exception as e:
            st.warning(f"⚠️ Impossible de charger le normalisateur : {e}")
    else:
        st.warning("⚠️ Fichier 'scaler.pkl' introuvable. La normalisation sera ignorée.")

    # ── Chargement des modèles .pkl (ARX, MLP, ANFIS) ──
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

    # ── Chargement des modèles .h5 (GRU, LSTM) ──
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
    """
    Charge un fichier Excel (.xlsx) ou CSV depuis un objet de type fichier.
    Retourne un DataFrame pandas ou None en cas d'erreur.
    """
    try:
        nom = getattr(source, "name", str(source))
        if nom.endswith(".csv"):
            df = pd.read_csv(source, sep=None, engine="python")
        else:
            df = pd.read_excel(source)
        return df
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}")
        return None


def verifier_colonnes_meteo(df: pd.DataFrame) -> bool:
    """
    Vérifie que le DataFrame contient les colonnes météo nécessaires.
    Affiche un message d'erreur pédagogique si une colonne est manquante.
    """
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
    """
    Applique la normalisation (StandardScaler ou MinMaxScaler) sur les
    colonnes météo. Retourne un tableau NumPy normalisé.
    """
    X = df[COLONNES_METEO].values.astype(float)
    if scaler is not None:
        try:
            X_norm = scaler.transform(X)
            return X_norm
        except Exception as e:
            st.warning(f"⚠️ Erreur lors de la normalisation : {e}. Les données brutes seront utilisées.")
    return X


def predire(modele, cle_modele: str, X_norm: np.ndarray) -> np.ndarray | None:
    """
    Appelle la méthode de prédiction appropriée selon le type de modèle.
    - Pour les modèles récurrents (GRU, LSTM), reshape les données en 3D.
    - Pour les autres modèles, utilise la forme 2D standard.
    Retourne un tableau NumPy des valeurs prédites.
    """
    try:
        if cle_modele in ("gru", "lstm"):
            # Les modèles récurrents attendent (nb_échantillons, nb_pas_temps, nb_features)
            X_3d = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
            predictions = modele.predict(X_3d, verbose=0).flatten()
        else:
            predictions = modele.predict(X_norm).flatten()
        return predictions
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction avec le modèle sélectionné : {e}")
        return None


def calculer_metriques(reelles: np.ndarray, predites: np.ndarray) -> dict:
    """
    Calcule les indicateurs de performance classiques :
    - RMSE : Racine de l'Erreur Quadratique Moyenne
    - MAE  : Erreur Absolue Moyenne
    - R²   : Coefficient de Détermination
    """
    rmse = np.sqrt(mean_squared_error(reelles, predites))
    mae  = mean_absolute_error(reelles, predites)
    r2   = r2_score(reelles, predites)
    return {"RMSE": rmse, "MAE": mae, "R²": r2}


def creer_graphique_comparaison(
    reelles: np.ndarray,
    predites: np.ndarray,
    nom_modele: str,
) -> go.Figure:
    """
    Génère un graphique Plotly interactif superposant la puissance réelle
    et la puissance prédite par le modèle sélectionné.
    """
    indices = list(range(len(reelles)))
    fig = go.Figure()

    # ── Courbe de la puissance réelle ──
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

    # ── Courbe de la puissance prédite ──
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
        xaxis=dict(
            title="Échantillon (indice temporel)",
            gridcolor="#222",
            color="#aaa",
        ),
        yaxis=dict(
            title="Puissance (kW)",
            gridcolor="#222",
            color="#aaa",
        ),
        plot_bgcolor="#0F0F1A",
        paper_bgcolor="#16213E",
        font=dict(color="#E8E8F0"),
        legend=dict(
            bgcolor="#1A1A2E",
            bordercolor="#FF6B2B",
            borderwidth=1,
        ),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def creer_graphique_prediction(predites: np.ndarray, nom_modele: str) -> go.Figure:
    """
    Génère un graphique Plotly n'affichant que la courbe de puissance prédite
    (pour la page de prédiction future où la puissance réelle est inconnue).
    """
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
        xaxis=dict(
            title="Horizon temporel (heures)",
            gridcolor="#222",
            color="#aaa",
        ),
        yaxis=dict(
            title="Puissance estimée (kW)",
            gridcolor="#222",
            color="#aaa",
        ),
        plot_bgcolor="#0F0F1A",
        paper_bgcolor="#16213E",
        font=dict(color="#E8E8F0"),
        legend=dict(bgcolor="#1A1A2E", bordercolor="#FF6B2B", borderwidth=1),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def afficher_carte_metrique(label: str, valeur: float, unite: str = "", precision: int = 4):
    """
    Affiche une carte HTML stylisée pour une métrique de performance.
    """
    st.markdown(
        f"""
        <div class="carte-metrique">
            <div class="valeur">{valeur:.{precision}f} <span style="font-size:1rem;color:#aaa">{unite}</span></div>
            <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. BARRE LATÉRALE — Navigation et informations globales
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Logo / En-tête ──
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

    # ── Navigation entre les pages ──
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

    # ── Sélection globale du modèle ──
    st.markdown("### 🤖 Modèle d'IA Actif")
    nom_modele_affiche = st.selectbox(
        label="Choisir un modèle :",
        options=list(MODELES_DISPONIBLES.keys()),
        label_visibility="collapsed",
    )
    cle_modele = MODELES_DISPONIBLES[nom_modele_affiche]
    nom_modele_court = nom_modele_affiche.split("—")[0].strip()

    st.divider()

    # ── Informations système ──
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

# ──────────────────────────────────────────────────────────────────────────────
# 4. CHARGEMENT DES MODÈLES (effectué une seule fois au démarrage)
# ──────────────────────────────────────────────────────────────────────────────

modeles, scaler = charger_modeles()

# ──────────────────────────────────────────────────────────────────────────────
# 5. ÉTAT DE SESSION — Persistance des données entre les pages
# ──────────────────────────────────────────────────────────────────────────────

if "donnees_chargees" not in st.session_state:
    st.session_state.donnees_chargees = None  # DataFrame principal

if "nom_fichier" not in st.session_state:
    st.session_state.nom_fichier = None


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████  PAGE 1 : IMPORTATION DES DONNÉES  ██████████████████████
# ══════════════════════════════════════════════════════════════════════════════
if page_choisie == "📂 Importation des Données":

    # ── En-tête de la page ──
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

    # ── Section 1 : Illustration du système de collecte ──
    with st.container():
        col_img, col_desc = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("#### 🖼️ Protocole de Collecte des Données")
            # Zone d'image illustrative — remplacez le chemin par votre image réelle
            if os.path.exists("schema_capteurs.png"):
                st.image(
                    "schema_capteurs.png",
                    caption="Schéma du dispositif de mesure météorologique",
                    use_container_width=True,
                )
            else:
                # Placeholder élégant si l'image n'est pas disponible
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

    # ── Section 2 : Chargement du fichier ──
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
                • <code>Éclairage</code><br/>
                • <code>Humidité</code><br/>
                • <code>Température</code><br/><br/>
                Colonne optionnelle (évaluation) :<br/>
                • <code>Puissance</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Logique de chargement ──
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
        # ── Chargement du fichier exemple par défaut ──
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

    # ── Aperçu des données ──
    if st.session_state.donnees_chargees is not None:
        df_affiche = st.session_state.donnees_chargees
        st.divider()
        st.markdown(
            f"#### 🔍 Aperçu des Données — `{st.session_state.nom_fichier}`"
        )

        # ── Statistiques rapides ──
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

        # ── Tableau des premières lignes ──
        st.markdown("**Premières lignes du tableau :**")
        st.dataframe(
            df_affiche.head(20),
            use_container_width=True,
            hide_index=False,
        )

        # ── Statistiques descriptives ──
        with st.expander("📈 Statistiques Descriptives (cliquez pour déplier)"):
            st.dataframe(df_affiche.describe().round(4), use_container_width=True)

        # ── Mini graphiques d'exploration ──
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


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████  PAGE 2 : ÉVALUATION DES MODÈLES  ███████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
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

    # ── Vérification préalable : données disponibles ──
    if st.session_state.donnees_chargees is None:
        st.warning(
            "⚠️ **Aucune donnée chargée.** "
            "Veuillez d'abord vous rendre sur la page **📂 Importation des Données** "
            "et charger un fichier."
        )
        st.stop()

    df = st.session_state.donnees_chargees

    # ── Vérification de la présence des colonnes météo ──
    if not verifier_colonnes_meteo(df):
        st.stop()

    # ── Vérification de la présence de la colonne puissance ──
    if COLONNE_PUISSANCE not in df.columns:
        st.error(
            f"❌ **La colonne '{COLONNE_PUISSANCE}' est absente** dans le fichier chargé.\n\n"
            "Cette page nécessite la puissance réelle mesurée pour effectuer la validation. "
            "Veuillez charger un fichier contenant cette colonne, ou utiliser la page "
            "**🔮 Prédiction Future** pour travailler sans données de référence."
        )
        st.stop()

    # ── Récupération du modèle sélectionné dans la barre latérale ──
    modele_actif = modeles.get(cle_modele)
    if modele_actif is None:
        st.error(
            f"❌ Le modèle **{nom_modele_court}** n'est pas disponible. "
            "Vérifiez que le fichier correspondant est présent dans votre dépôt."
        )
        st.stop()

    # ── Normalisation des données ──
    with st.spinner("⚙️ Normalisation des données météorologiques…"):
        X_norm = normaliser_donnees(df, scaler)

    # ── Prédiction ──
    with st.spinner(f"🤖 Calcul des prédictions avec le modèle {nom_modele_court}…"):
        y_predit = predire(modele_actif, cle_modele, X_norm)

    if y_predit is None:
        st.stop()

    # ── Récupération de la puissance réelle ──
    y_reel = df[COLONNE_PUISSANCE].values.astype(float)

    # ── Alignement des longueurs (au cas où le modèle retournerait moins de points) ──
    n_min    = min(len(y_reel), len(y_predit))
    y_reel   = y_reel[:n_min]
    y_predit = y_predit[:n_min]

    # ── Calcul des métriques ──
    metriques = calculer_metriques(y_reel, y_predit)

    # ── Affichage des métriques ──
    st.markdown(f"### 📐 Indicateurs de Performance — {nom_modele_court}")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        afficher_carte_metrique("RMSE — Erreur Quadratique Moyenne", metriques["RMSE"], "kW")
    with col_m2:
        afficher_carte_metrique("MAE — Erreur Absolue Moyenne",      metriques["MAE"],  "kW")
    with col_m3:
        afficher_carte_metrique("R² — Coefficient de Détermination", metriques["R²"],  "",  4)

    # ── Interprétation automatique du R² ──
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

    # ── Graphique de comparaison ──
    st.markdown("### 📈 Graphique de Validation")
    fig_comparaison = creer_graphique_comparaison(y_reel, y_predit, nom_modele_court)
    st.plotly_chart(fig_comparaison, use_container_width=True)

    # ── Graphique de parité (nuage de points Réel vs Prédit) ──
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
        # Droite de parité parfaite
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

    # ── Tableau de comparaison numérique ──
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

    # ── Export des résultats ──
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


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  PAGE 3 : PRÉDICTION FUTURE  ███████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
elif page_choisie == "🔮 Prédiction Future":

    st.markdown(
        """
        <h1 style="text-align:center;">🔮 Prédiction Pure & Exploitation</h1>
        <p style="text-align:center; color:#aaa; font-size:1rem;">
            Estimez la production d'énergie future sans données de puissance réelle.
        </p>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    # ── Vérification du modèle ──
    modele_actif = modeles.get(cle_modele)
    if modele_actif is None:
        st.error(
            f"❌ Le modèle **{nom_modele_court}** n'est pas disponible. "
            "Vérifiez que le fichier correspondant est présent dans votre dépôt."
        )
        st.stop()

    # ── Sélection du mode de prédiction ──
    mode_prediction = st.radio(
        label="Mode de prédiction :",
        options=[
            "🎛️  Saisie Manuelle — Prédiction instantanée",
            "📁  Chargement de Prévisions — Prédiction journalière",
        ],
        horizontal=True,
    )

    st.divider()

    # ══════════════════════════════════════════════════════
    # MODE 1 : SAISIE MANUELLE
    # ══════════════════════════════════════════════════════
    if mode_prediction.startswith("🎛️"):

        st.markdown(f"#### 🎛️ Saisie Manuelle des Conditions Météorologiques")
        st.markdown(
            f"Ajustez les curseurs ci-dessous pour simuler un instant météorologique "
            f"et obtenir la puissance estimée par le modèle **{nom_modele_court}**."
        )

        col_sl1, col_sl2, col_sl3 = st.columns(3)
        with col_sl1:
            valeur_temperature = st.slider(
                label="🌡️ Température (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=25.0,
                step=0.5,
                format="%.1f °C",
            )
        with col_sl2:
            valeur_humidite = st.slider(
                label="💧 Humidité (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                format="%.0f %%",
            )
        with col_sl3:
            valeur_eclairage = st.slider(
                label="☀️ Éclairage (lux)",
                min_value=0.0,
                max_value=120000.0,
                value=50000.0,
                step=500.0,
                format="%.0f lux",
            )

        st.markdown(" ")

        # ── Calcul en temps réel ──
        X_manuel = np.array([[valeur_eclairage, valeur_humidite, valeur_temperature]])
        X_norm_manuel = X_manuel.copy()
        if scaler is not None:
            try:
                X_norm_manuel = scaler.transform(X_manuel)
            except Exception:
                pass

        puissance_estimee = predire(modele_actif, cle_modele, X_norm_manuel)

        if puissance_estimee is not None:
            val_kw = float(puissance_estimee[0])
            # ── Résultat affiché dans une carte visuelle ──
            st.markdown(
                f"""
                <div class="resultat-prediction">
                    <p style="color:#aaa; margin-bottom:8px; font-size:0.9rem;">
                        ⚡ Puissance estimée par le modèle <b>{nom_modele_court}</b>
                    </p>
                    <div class="puissance">{val_kw:.4f} <span class="unite">kW</span></div>
                    <p style="color:#888; margin-top:12px; font-size:0.8rem;">
                        Conditions : {valeur_temperature:.1f}°C · {valeur_humidite:.0f}% HR · {valeur_eclairage:.0f} lux
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Jauge visuelle ──
            val_max_estime = max(val_kw * 2, 1.0)  # Borne supérieure dynamique
            fig_jauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=val_kw,
                    number=dict(suffix=" kW", font=dict(size=28, color="#FFC107")),
                    gauge=dict(
                        axis=dict(range=[0, val_max_estime], tickcolor="#aaa"),
                        bar=dict(color="#FF6B2B"),
                        bgcolor="#16213E",
                        bordercolor="#333",
                        steps=[
                            dict(range=[0, val_max_estime * 0.33], color="#1A1A2E"),
                            dict(range=[val_max_estime * 0.33, val_max_estime * 0.66], color="#1E2A3A"),
                            dict(range=[val_max_estime * 0.66, val_max_estime], color="#243040"),
                        ],
                        threshold=dict(
                            line=dict(color="#FFC107", width=3),
                            thickness=0.85,
                            value=val_kw,
                        ),
                    ),
                    title=dict(
                        text="Puissance Instantanée Estimée",
                        font=dict(size=14, color="#aaa"),
                    ),
                )
            )
            fig_jauge.update_layout(
                paper_bgcolor="#16213E",
                font_color="#E8E8F0",
                height=280,
                margin=dict(l=40, r=40, t=40, b=20),
            )
            st.plotly_chart(fig_jauge, use_container_width=True)

    # ══════════════════════════════════════════════════════
    # MODE 2 : CHARGEMENT DE PRÉVISIONS MÉTÉO
    # ══════════════════════════════════════════════════════
    else:

        st.markdown("#### 📁 Chargement d'un Fichier de Prévisions Météorologiques")
        st.markdown(
            "Importez un fichier Excel ou CSV contenant les colonnes "
            f"**{', '.join(COLONNES_METEO)}** "
            "(sans colonne de puissance) pour obtenir la courbe de production prévue."
        )

        fichier_previsions = st.file_uploader(
            label="Sélectionner le fichier de prévisions :",
            type=["xlsx", "xls", "csv"],
            key="uploader_previsions",
            help=(
                "Ce fichier ne doit contenir que les variables météo : "
                f"{', '.join(COLONNES_METEO)}"
            ),
        )

        if fichier_previsions is not None:
            df_prev = charger_donnees(fichier_previsions)
            if df_prev is not None and verifier_colonnes_meteo(df_prev):

                st.success(
                    f"✅ Fichier de prévisions chargé : "
                    f"**{fichier_previsions.name}** — {len(df_prev)} horizons temporels."
                )
                st.dataframe(df_prev.head(10), use_container_width=True)

                # ── Normalisation ──
                with st.spinner("⚙️ Normalisation des données de prévision…"):
                    X_prev_norm = normaliser_donnees(df_prev, scaler)

                # ── Prédiction ──
                with st.spinner(f"🔮 Calcul de la prévision de production…"):
                    y_prev = predire(modele_actif, cle_modele, X_prev_norm)

                if y_prev is not None:
                    # ── Graphique de prévision ──
                    st.divider()
                    st.markdown(
                        f"### 📈 Prévision de Production — Modèle {nom_modele_court}"
                    )
                    fig_prev = creer_graphique_prediction(y_prev, nom_modele_court)
                    st.plotly_chart(fig_prev, use_container_width=True)

                    # ── Statistiques de la prévision ──
                    col_p1, col_p2, col_p3 = st.columns(3)
                    with col_p1:
                        afficher_carte_metrique("Puissance Maximale",  float(y_prev.max()),  "kW")
                    with col_p2:
                        afficher_carte_metrique("Puissance Moyenne",   float(y_prev.mean()), "kW")
                    with col_p3:
                        afficher_carte_metrique("Énergie Totale Estimée",
                                                float(y_prev.sum()),  "kWh")

                    # ── Export de la prévision ──
                    st.divider()
                    df_sortie = df_prev.copy()
                    df_sortie["Puissance_Predite_kW"] = y_prev
                    buffer_prev = io.BytesIO()
                    df_sortie.to_excel(buffer_prev, index=False)
                    st.download_button(
                        label="📥 Télécharger la prévision (Excel)",
                        data=buffer_prev.getvalue(),
                        file_name=f"prevision_{cle_modele}.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument"
                            ".spreadsheetml.sheet"
                        ),
                    )

        else:
            # ── Message d'invite lorsque aucun fichier n'est chargé ──
            st.markdown(
                """
                <div style="
                    background: #1A1A2E;
                    border: 2px dashed #FF6B2B44;
                    border-radius: 12px;
                    padding: 50px 30px;
                    text-align: center;
                    color: #555;
                    margin-top: 20px;
                ">
                    <div style="font-size:3rem;">📂</div>
                    <p style="margin-top:12px; font-size:1rem; color:#888;">
                        Importez un fichier de prévisions météorologiques<br/>
                        pour visualiser la courbe de production estimée.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# 6. PIED DE PAGE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="margin-top:40px;"/>
    <div style="text-align:center; color:#555; font-size:0.78rem; padding: 10px 0 20px 0;">
        ☀️ Plateforme de Prédiction d'Énergie Photovoltaïque par IA &nbsp;|&nbsp;
        Projet de Fin d'Études — Ingénierie des Systèmes &amp; Automatique &nbsp;|&nbsp;
        Développé avec <a href="https://streamlit.io" style="color:#FF6B2B;">Streamlit</a>
    </div>
    """,
    unsafe_allow_html=True,
)
