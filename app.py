"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PLATEFORME DE PRÉDICTION D'ÉNERGIE PHOTOVOLTAÏQUE PAR IA            ║
║                Projet de Fin d'Études — Ingénierie des Systèmes              ║
╚══════════════════════════════════════════════════════════════════════════════╝
Auteurs  : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI
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

# Configuration des colonnes physiques des capteurs
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

# Chargement mis en cache des fichiers de modèles d'IA
@st.cache_resource(show_spinner="⚙️ Chargement des architectures d'IA…")
def charger_ressources():
    modeles = {}
    scaler = None

    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
        except Exception as e:
            st.warning(f"ℹ️ Info Scaler : {e}")

    fichiers_pkl = {
        "arx": "model_arx.pkl",
        "mlp": "model_mlp.pkl",
        "anfis": "model_anfis.pkl",
    }
    
    for cle, chemin in fichiers_pkl.items():
        if os.path.exists(chemin):
            try:
                modeles[cle] = joblib.load(chemin)
            except Exception as e:
                st.error(f"❌ Erreur sur {chemin} : {e}")

    try:
        from tensorflow.keras.models import load_model
        fichiers_h5 = {
            "gru": "model_gru.h5",
            "lstm": "model_lstm.h5",
        }
        for cle, chemin in fichiers_h5.items():
            if os.path.exists(chemin):
                try:
                    modeles[cle] = load_model(chemin, compile=False)
                except Exception as e:
                    st.error(f"❌ Erreur sur {chemin} : {e}")
    except ImportError:
        pass

    return modeles, scaler

def preparer_matrice_entrees(df_data: pd.DataFrame, scaler_obj) -> np.ndarray:
    X_base = df_data[COLONNES_REQUISES].values.astype(float)
    if np.any(np.isnan(X_base)):
        X_base = np.nan_to_num(X_base, nan=0.0)

    if scaler_obj is not None:
        try:
            # Tente d'appliquer le scaler sur l'ensemble ou s'adapte s'il n'attend qu'une colonne
            if hasattr(scaler_obj, "n_features_in_") and scaler_obj.n_features_in_ == 1:
                X_base = scaler_obj.transform(X_base[:, [0]])
            else:
                X_base = scaler_obj.transform(X_base)
        except Exception:
            pass
            
    return X_base

modeles, scaler = charger_ressources()

if "donnees" not in st.session_state:
    st.session_state.donnees = None
if "nom_fichier" not in st.session_state:
    st.session_state.nom_fichier = None

# Menu latéral de navigation
with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 10px 0;'><div style='font-size:3rem;'>☀️</div></div>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Menu Principal")
    page = st.radio(
        "Sélectionnez une page :", 
        ["🏠 Accueil & Présentation", "📂 Importation des Données", "📊 Évaluation & Graphiques", "🔮 Prédiction Future"]
    )
    st.divider()
    st.markdown("### 🤖 Configuration IA")
    nom_modele_selectionne = st.selectbox("Modèle d'IA actif :", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele_selectionne]
    nom_court = nom_modele_selectionne.split("—")[0].strip()

# ── PAGE 1 : ACCUEIL & PRÉSENTATION COMPLÈTE RESTAURÉE ──
if page == "🏠 Accueil & Présentation":
    if os.path.exists("panneau_pv.jpg"):
        st.image("panneau_pv.jpg", caption="Dispositif expérimental d'acquisition", use_container_width=True)
    
    st.markdown(
        """
        <div style="text-align: center; margin-top: 25px; margin-bottom: 25px;">
            <span class="badge-pfe">PROJET DE FIN D'ÉTUDES (PFE)</span>
            <h1 style="font-size: 2.3rem;">Prédiction de la Production d'Énergie Photovoltaïque par Intelligence Artificielle</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col_gauche, col_droite = st.columns([1, 1], gap="large")
    with col_gauche:
        st.markdown("### 📝 Fiche Technique du Projet")
        st.markdown(
            """
            <div class="cadre-accueil">
                <h4 style="color:#E65100; margin-top:0; font-size:1.15rem;">👥 Réalisé par :</h4>
                <ul style="font-size: 1.05rem; line-height: 1.7; margin-bottom: 15px;">
                    <li><b>Nor El Houda BEKADA BENCHAIB</b></li>
                    <li><b>Yousra Oum El kheir HAMMADI</b></li>
                </ul>
                <hr style="margin: 15px 0; border-color: #DEE2E6;"/>
                <h4 style="color:#E65100; font-size:1.15rem;">👨‍🏫 Encadré par :</h4>
                <p style="font-size: 1.05rem; margin-bottom: 5px;"><b>M. Anisse CHIALI</b></p>
                <p style="font-size: 1.05rem;"><b>Mme Imane NEDJAR</b></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col_droite:
        st.markdown("### 💡 À propos de cette application")
        st.markdown(
            """
            Cette plateforme logicielle a été conçue pour automatiser l'analyse, le traitement du signal et l'évaluation 
            de différents modèles mathématiques et d'intelligence artificielle appliqués à la prédiction énergétique.
            
            Elle intègre des modèles d'ingénierie avancés permettant de confronter les approches linéaires classiques 
            (ARX) aux techniques d'apprentissage profond (PMC, GRU, LSTM) ainsi qu'aux systèmes flous (ANFIS).
            """
        )

# ── PAGE 2 : IMPORTATION DES DONNÉES ──
elif page == "📂 Importation des Données":
    st.title("📂 Importation des Données Capteurs")
    fichier_charge = st.file_uploader("Téléverser votre fichier Excel ou CSV :", type=["xlsx", "xls", "csv"])
    
    if fichier_charge is not None:
        try:
            if fichier_charge.name.endswith(".csv"):
                df_input = pd.read_csv(fichier_charge, sep=None, engine="python")
            else:
                df_input = pd.read_excel(fichier_charge)
            st.session_state.donnees = df_input
            st.session_state.nom_fichier = fichier_charge.name
            st.success(f"✅ Fichier '{fichier_charge.name}' chargé !")
        except Exception as e:
            st.error(f"❌ Erreur de lecture : {e}")
    elif st.session_state.donnees is None and os.path.exists(FICHIER_EXEMPLE):
        st.session_state.donnees = pd.read_excel(FICHIER_EXEMPLE)
        st.session_state.nom_fichier = FICHIER_EXEMPLE

    if st.session_state.donnees is not None:
        st.markdown(f"### Premières lignes du fichier : `{st.session_state.nom_fichier}`")
        st.dataframe(st.session_state.donnees.head(15), use_container_width=True)

# ── PAGE 3 : ÉVALUATION ET GRAPHES (CORRIGÉE DES DIMENSIONS) ──
elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Traitement & Évaluation des Modèles")
    if st.session_state.donnees is None:
        st.warning("⚠️ Veuillez d'abord importer un fichier de données.")
        st.stop()
        
    df = st.session_state.donnees.copy().dropna(subset=COLONNES_REQUISES + [COLONNE_CIBLE])
    
    obj_modele = modeles.get(cle_modele)
    if obj_modele is None:
        st.error(f"❌ Le modèle binaire '{nom_court}' est introuvable sur votre instance.")
        st.stop()

    X_final = preparer_matrice_entrees(df, scaler)
    
    try:
        if cle_modele in ["gru", "lstm"]:
            # Vérification de la dimension d'entrée attendue par la première couche du modèle Keras
            shape_attendue = obj_modele.input_shape
            nb_features_attendues = shape_attendue[-1] if shape_attendue else 3
            
            if nb_features_attendues == 1:
                # Si le modèle récurrent n'attend qu'une variable (LDR_Raw uniquement)
                X_input_rnn = np.reshape(X_final[:, 0], (X_final.shape[0], 1, 1))
            else:
                # Si le modèle attend les 3 variables [Echantillons, Pas de temps=1, Features=3]
                X_input_rnn = np.reshape(X_final, (X_final.shape[0], 1, X_final.shape[1]))
                
            y_pred = obj_modele.predict(X_input_rnn, verbose=0).flatten()
        else:
            y_pred = obj_modele.predict(X_final).flatten()
            
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul mathématique dimensionnel : {e}")
        st.stop()

    y_reel = df[COLONNE_CIBLE].values.astype(float)
    
    taille_commune = min(len(y_reel), len(y_pred))
    y_reel = y_reel[:taille_commune]
    y_pred = y_pred[:taille_commune]

    rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
    mae  = mean_absolute_error(y_reel, y_pred)
    r2   = r2_score(y_reel, y_pred)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="carte-metrique"><div class="valeur">{rmse:.2f}</div><div class="label">RMSE (mW)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="carte-metrique"><div class="valeur">{mae:.2f}</div><div class="label">MAE (mW)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="carte-metrique"><div class="valeur">{r2:.4f}</div><div class="label">R² (Score Global)</div></div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_reel, name="⚡ Valeur Réelle Mesurée", mode="lines", line=dict(color="#1A73E8")))
    fig.add_trace(go.Scatter(y=y_pred, name=f"🤖 Prédiction {nom_court}", mode="lines", line=dict(color="#FF6B2B", dash="dash")))
    fig.update_layout(
        title="Validation croisée : puissance réelle mesurée vs puissance prédite par l'IA",
        xaxis_title="Points d'échantillonnage",
        yaxis_title="Puissance Électrique (mW)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 4 : PRÉDICTION FUTURE INTERACTIVE ──
elif page == "🔮 Prédiction Future":
    st.title("🔮 Simulation sur un Horizon Temporel")
    
    obj_modele = modeles.get(cle_modele)
    if obj_modele is None:
        st.error("❌ Modèle non chargé.")
        st.stop()

    mode_test = st.radio(
        "Choisissez votre mode d'analyse :", 
        ["🎛️ Curseurs Dynamiques", "📈 Projections Graphiques"]
    )
    
    if mode_test == "🎛️ Curseurs Dynamiques":
        c1, c2, c3 = st.columns(3)
        with c1: val_ldr = st.slider("Éclairement (LDR_Raw)", 0, 4095, 1500)
        with c2: val_hum = st.slider("Humidité (Hum_%)", 0.0, 100.0, 65.0)
        with c3: val_temp = st.slider("Température (Temp_C)", -5.0, 50.0, 25.0)
        
        df_temp = pd.DataFrame([{"LDR_Raw": val_ldr, "Hum_%": val_hum, "Temp_C": val_temp}])
        X_pt = preparer_matrice_entrees(df_temp, scaler)
        
        if cle_modele in ["gru", "lstm"]:
            shape_attendue = obj_modele.input_shape
            if shape_attendue and shape_attendue[-1] == 1:
                X_pt = np.reshape(X_pt[:, 0], (1, 1, 1))
            else:
                X_pt = np.reshape(X_pt, (1, 1, 3))
            
        pred = obj_modele.predict(X_pt, verbose=0).flatten()[0]
        st.metric(label=f"⚡ Puissance Estimée (Horizon Instantané) par {nom_court}", value=f"{max(0.0, pred):.2f} mW")

    elif mode_test == "📈 Projections Graphiques":
        if st.session_state.donnees is None:
            st.warning("⚠️ Importez un fichier pour projeter les courbes.")
            st.stop()
            
        horizon = st.slider("Nombre de points futurs à simuler :", 5, 100, 30)
        df_horizon = st.session_state.donnees.copy().head(horizon)
        X_hor = preparer_matrice_entrees(df_horizon, scaler)
        
        if cle_modele in ["gru", "lstm"]:
            shape_attendue = obj_modele.input_shape
            if shape_attendue and shape_attendue[-1] == 1:
                X_hor = np.reshape(X_hor[:, 0], (X_hor.shape[0], 1, 1))
            else:
                X_hor = np.reshape(X_hor, (X_hor.shape[0], 1, 3))
            
        preds_hor = obj_modele.predict(X_hor, verbose=0).flatten()
        
        fig_futur = go.Figure()
        fig_futur.add_trace(go.Scatter(y=np.clip(preds_hor, 0, None), mode="lines+markers", name=f"Horizon Prévisionnel ({nom_court})", line=dict(color="#E65100", width=3)))
        fig_futur.update_layout(title="Horizon Prévisionnel de Puissance Électrique", xaxis_title="Points temporels cumulés", yaxis_title="Puissance Prédite (mW)")
        st.plotly_chart(fig_futur, use_container_width=True)
