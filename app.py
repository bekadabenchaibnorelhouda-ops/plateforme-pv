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

# Configuration des colonnes physiques de votre installation
COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C"]
COLONNE_CIBLE     = "Puissance_mW"
FICHIER_EXEMPLE   = "Classeur1.xlsx"

MODELES_DISPONIBLES = {
    "💾 LSTM (Long Short-Term Memory)"         : "lstm",
    "🔁 GRU (Gated Recurrent Unit)"            : "gru",
    "🧠 MLP (Multi-Layer Perceptron)"         : "mlp",
    "📐 ARX (Auto-Regressive Exogenous)"       : "arx",
    "🔮 ANFIS (Adaptive Neuro-Fuzzy)"          : "anfis",
}

# Chargement intelligent des modèles d'IA et du scaler
@st.cache_resource(show_spinner="⚙️ Chargement des architectures d'IA…")
def charger_ressources():
    modeles = {}
    scaler = None

    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
        except:
            pass

    fichiers_pkl = {
        "mlp": "model_mlp.pkl",
        "arx": "model_arx.pkl",
        "anfis": "model_anfis.pkl",
    }
    
    for cle, chemin in fichiers_pkl.items():
        if os.path.exists(chemin):
            try:
                modeles[cle] = joblib.load(chemin)
            except:
                pass

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
                except:
                    pass
    except ImportError:
        pass

    return modeles, scaler

modeles, scaler = charger_ressources()

# Initialisation des variables d'état de session Streamlit
if "donnees" not in st.session_state:
    st.session_state.donnees = None
if "nom_fichier" not in st.session_state:
    st.session_state.nom_fichier = None

# ==========================================
# BARRE LATÉRALE (Navigation Origine)
# ==========================================
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
    nom_court = nom_modele_selectionne.split("(")[0].strip()

# ==========================================
# ── PAGE 1 : ACCUEIL & PRÉSENTATION ──
# ==========================================
if page == "🏠 Accueil & Présentation":
    if os.path.exists("panneau_pv.jpg"):
        st.image("panneau_pv.jpg", caption="Dispositif expérimental d'acquisition de données", use_container_width=True)
    
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
            aux techniques d'apprentissage profond ainsi qu'aux systèmes flous à partir de vos capteurs physiques.
            """
        )

# ==========================================
# ── PAGE 2 : IMPORTATION DES DONNÉES ──
# ==========================================
elif page == "📂 Importation des Données":
    st.title("📂 Importation des Données Capteurs")
    st.write("Chargez votre fichier de mesures pour alimenter et calibrer le moteur de prédiction de la plateforme.")
    
    fichier_charge = st.file_uploader("Choisir votre fichier (Excel ou CSV) :", type=["xlsx", "xls", "csv"])
    
    if fichier_charge is not None:
        try:
            if fichier_charge.name.endswith(".csv"):
                df_input = pd.read_csv(fichier_charge, sep=None, engine="python")
            else:
                df_input = pd.read_excel(fichier_charge)
            
            # Nettoyage et sécurité anti-bug Arrow pour l'affichage des colonnes temporelles
            df_cleaned = df_input.copy()
            if "Date" in df_cleaned.columns: df_cleaned["Date"] = df_cleaned["Date"].astype(str)
            if "Heure" in df_cleaned.columns: df_cleaned["Heure"] = df_cleaned["Heure"].astype(str)
            
            st.session_state.donnees = df_cleaned
            st.session_state.nom_fichier = fichier_charge.name
            st.success(f"✅ Fichier '{fichier_charge.name}' chargé et mémorisé avec succès !")
        except Exception as e:
            st.error(f"❌ Erreur lors du décodage du fichier : {e}")
            
    elif st.session_state.donnees is None and os.path.exists(FICHIER_EXEMPLE):
        df_ex = pd.read_excel(FICHIER_EXEMPLE)
        if "Date" in df_ex.columns: df_ex["Date"] = df_ex["Date"].astype(str)
        if "Heure" in df_ex.columns: df_ex["Heure"] = df_ex["Heure"].astype(str)
        st.session_state.donnees = df_ex
        st.session_state.nom_fichier = FICHIER_EXEMPLE

    if st.session_state.donnees is not None:
        st.markdown(f"### 📊 Aperçu des données : `{st.session_state.nom_fichier}`")
        st.dataframe(st.session_state.donnees.head(10), use_container_width=True)
        st.markdown("### 📈 Statistiques descriptives du fichier")
        st.write(st.session_state.donnees.describe())
    else:
        st.info("ℹ️ En attente de l'importation d'un fichier utilisateur.")

# ==========================================
# ── PAGE 3 : ÉVALUATION ET GRAPHES (MÉTRIQUES RÉPARÉES) ──
# ==========================================
elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Traitement & Évaluation Réelle des Modèles")
    if st.session_state.donnees is None:
        st.warning("⚠️ Attention : Veuillez d'abord charger un fichier de données dans l'onglet '📂 Importation des Données'.")
        st.stop()
        
    df_global = st.session_state.donnees.copy().dropna(subset=COLONNES_REQUISES + [COLONNE_CIBLE])
    
    st.markdown("### ⚙️ Choix de l'ensemble d'analyse (Validation Croisée)")
    choix_split = st.radio(
        "Pour obtenir les métriques exactes de votre code piéton, sélectionnez la portion de Test :", 
        ["📈 Ensemble de Test (Derniers 15% du fichier)", "📊 Fichier Global Complet (100%)"]
    )
    
    total_lignes = len(df_global)
    idx_val = int(total_lignes * 0.85)
    
    if choix_split == "📈 Ensemble de Test (Derniers 15% du fichier)":
        df = df_global.iloc[idx_val:].reset_index(drop=True)
    else:
        df = df_global.reset_index(drop=True)

    obj_modele = modeles.get(cle_modele)
    if obj_modele is None:
        st.error(f"❌ Le fichier binaire du modèle '{nom_court}' est introuvable dans votre répertoire actuel.")
        st.stop()

    # --- SÉCURITÉ ET ADAPTATION TECHNIQUE DE LA FORME DES COMPOSANTS (FEATURES) ---
    nb_features_attendues = 3
    if cle_modele in ["gru", "lstm"]:
        if hasattr(obj_modele, "input_shape") and obj_modele.input_shape is not None:
            nb_features_attendues = obj_modele.input_shape[-1]
    elif cle_modele in ["arx", "anfis"]:
        nb_features_attendues = 1  
    elif hasattr(obj_modele, "n_features_in_"):
        nb_features_attendues = obj_modele.n_features_in_

    # Extraction et standardisation adéquate des données
    if nb_features_attendues == 1:
        X_base = df[["LDR_Raw"]].values.astype(float)
        if scaler is not None:
            try:
                if hasattr(scaler, "mean_") and len(scaler.mean_) >= 1:
                    X_base = (X_base - scaler.mean_[0]) / np.sqrt(scaler.var_[0])
                else:
                    X_base = scaler.transform(X_base)
            except: pass
    else:
        X_base = df[COLONNES_REQUISES].values.astype(float)
        if scaler is not None:
            try: X_base = scaler.transform(X_base)
            except: pass

    X_base = np.nan_to_num(X_base, nan=0.0)
    y_reel = df[COLONNE_CIBLE].values.astype(float)

    # Inférence et calcul
    try:
        if cle_modele in ["gru", "lstm"]:
            X_input_rnn = np.reshape(X_base, (X_base.shape[0], 1, X_base.shape[1]))
            y_pred = obj_modele.predict(X_input_rnn, verbose=0).flatten()
        else:
            y_pred = obj_modele.predict(X_base).flatten()
            
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
    except Exception as e:
        st.error(f"❌ Erreur lors de l'exécution mathématique de {nom_court} : {e}")
        st.stop()

    taille_commune = min(len(y_reel), len(y_pred))
    y_reel = y_reel[:taille_commune]
    y_pred = y_pred[:taille_commune]

    # Calcul des vrais indicateurs statistiques
    rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
    mae  = mean_absolute_error(y_reel, y_pred)
    r2   = r2_score(y_reel, y_pred)

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="carte-metrique"><div class="valeur">{rmse:.2f}</div><div class="label">RMSE (mW)</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="carte-metrique"><div class="valeur">{mae:.2f}</div><div class="label">MAE (mW)</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="carte-metrique"><div class="valeur">{r2:.4f}</div><div class="label">R² (Coefficient de Détermination)</div></div>', unsafe_allow_html=True)

    # Visualisation interactive avec Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_reel[:200], name="⚡ Valeur Réelle Mesurée", mode="lines", line=dict(color="#1A73E8")))
    fig.add_trace(go.Scatter(y=y_pred[:200], name=f"🤖 Prédiction {nom_court}", mode="lines", line=dict(color="#FF6B2B", dash="dash")))
    fig.update_layout(
        title=f"Validation Croisée Temporelle — Modèle : {nom_court}",
        xaxis_title="Points d'échantillonnage (200 premiers points)",
        yaxis_title="Puissance Électrique (mW)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ── PAGE 4 : PRÉDICTION FUTURE INTERACTIVE ──
# ==========================================
elif page == "🔮 Prédiction Future":
    st.title("🔮 Espace de Prédiction Future Interactive")
    
    obj_modele = modeles.get(cle_modele)
    if obj_modele is None:
        st.error("❌ Modèle non opérationnel.")
        st.stop()

    mode_test = st.radio("Sélectionnez votre mode d'analyse :", ["🎛️ Curseurs (Sliders)", "📈 Projections Temporelles"])
    
    nb_features_attendues = 1 if cle_modele in ["arx", "anfis"] else 3
    if hasattr(obj_modele, "n_features_in_") and cle_modele == "mlp":
        nb_features_attendues = obj_modele.n_features_in_
        
    if mode_test == "🎛️ Curseurs (Sliders)":
        c1, c2, c3 = st.columns(3)
        with c1: val_ldr = st.slider("Éclairement (LDR_Raw)", 0, 4095, 1500)
        with c2: val_hum = st.slider("Humidité (Hum_%)", 0.0, 100.0, 65.0)
        with c3: val_temp = st.slider("Température (Temp_C)", -5.0, 50.0, 25.0)
        
        if nb_features_attendues == 1:
            X_pt = np.array([[val_ldr]], dtype=float)
            if scaler is not None:
                try:
                    if hasattr(scaler, "mean_") and len(scaler.mean_) >= 3:
                        X_pt = (X_pt - scaler.mean_[0]) / np.sqrt(scaler.var_[0])
                except: pass
        else:
            df_temp = pd.DataFrame([{"LDR_Raw": val_ldr, "Hum_%": val_hum, "Temp_C": val_temp}])
            X_pt = df_temp[COLONNES_REQUISES].values.astype(float)
            if scaler is not None:
                try: X_pt = scaler.transform(X_pt)
                except: pass

        if cle_modele in ["gru", "lstm"]:
            X_pt = np.reshape(X_pt, (1, 1, X_pt.shape[1]))
            
        pred = obj_modele.predict(X_pt, verbose=0).flatten()[0]
        st.metric(label=f"⚡ Puissance Estimée par l'IA ({nom_court})", value=f"{max(0.0, pred):.2f} mW")

    elif mode_test == "📈 Projections Temporelles":
        if st.session_state.donnees is None:
            st.warning("⚠️ Veuillez importer un fichier de données au préalable.")
            st.stop()
            
        horizon = st.slider("Nombre de points temporels à projeter :", 5, 150, 50)
        df_horizon = st.session_state.donnees.copy().head(horizon)
        
        if nb_features_attendues == 1:
            X_hor = df_horizon[["LDR_Raw"]].values.astype(float)
            if scaler is not None:
                try:
                    if hasattr(scaler, "mean_") and len(scaler.mean_) >= 3:
                        X_hor = (X_hor - scaler.mean_[0]) / np.sqrt(scaler.var_[0])
                except: pass
        else:
            X_hor = df_horizon[COLONNES_REQUISES].values.astype(float)
            if scaler is not None:
                try: X_hor = scaler.transform(X_hor)
                except: pass
                
        if cle_modele in ["gru", "lstm"]:
            X_hor = np.reshape(X_hor, (X_hor.shape[0], 1, X_hor.shape[1]))
            
        preds_hor = obj_modele.predict(X_hor, verbose=0).flatten()
        
        fig_futur = go.Figure()
        fig_futur.add_trace(go.Scatter(y=np.clip(preds_hor, 0, None), mode="lines+markers", name=f"Projection {nom_court}", line=dict(color="#E65100", width=3)))
        fig_futur.update_layout(title="Évolution de la puissance estimée sur l'horizon choisi", xaxis_title="Temps cumulé (+t)", yaxis_title="Puissance (mW)")
        st.plotly_chart(fig_futur, use_container_width=True)
