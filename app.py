"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PLATEFORME DE PRÉDICTION D'ÉNERGIE PHOTOVOLTAÏQUE PAR IA            ║
║                Projet de Fin d'Études — Ingénierie des Systèmes              ║
╚══════════════════════════════════════════════════════════════════════════════╝
Fichier  : app.py
Auteurs  : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI
Encadrement : M. Anisse CHIALI & Mme Imane NEDJAR
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

import plotly.graph_objects as go

# 1. Configuration de la page (Doit être la toute première commande)
st.set_page_config(
    page_title="Prédiction PV par IA",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Design, Thème Clair et Ajustements CSS
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
    
    /* Cartes de métriques */
    .carte-metrique {
        background-color: var(--couleur-carte); border: 1px solid #CED4DA; border-radius: 12px;
        padding: 20px 24px; text-align: center; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); margin-bottom: 12px;
    }
    .carte-metrique .valeur { font-size: 2rem; font-weight: 700; color: var(--couleur-primaire); }
    .carte-metrique .label { font-size: 0.85rem; color: #495057; margin-top: 4px; text-transform: uppercase; }
    
    /* Styles page d'accueil */
    .cadre-accueil {
        background-color: #F8F9FA; border: 1px solid #DEE2E6; border-radius: 16px; padding: 25px; margin-bottom: 20px;
    }
    .badge-pfe {
        background-color: #E65100; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; display: inline-block; margin-bottom: 15px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, var(--couleur-primaire), #FF8C42); color: white;
        border: none; border-radius: 8px; font-weight: 600; padding: 10px 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constantes pour la correspondance des colonnes Excel de vos capteurs
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

# 3. Chargement des fichiers d'IA avec correction d'orthographe (model_...)
@st.cache_resource(show_spinner="⚙️ Chargement des architectures d'IA…")
def charger_ressources():
    modeles = {}
    scaler = None

    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
        except Exception as e:
            st.warning(f"ℹ️ Info Scaler : {e}")

    # Utilisation stricte des noms de fichiers réels (model_...)
    fichiers = {
        "arx": "model_arx.pkl",
        "mlp": "model_mlp.pkl",
        "anfis": "model_anfis.pkl",
    }
    
    for cle, chemin in fichiers.items():
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

def preparer_matrice_entrees(df_data: pd.DataFrame, scaler_obj, cle_mod, nb_attendues) -> np.ndarray:
    X_base = df_data[COLONNES_REQUISES].values.astype(float)
    if np.any(np.isnan(X_base)):
        X_base = np.nan_to_num(X_base, nan=0.0)

    if scaler_obj is not None:
        try:
            if hasattr(scaler_obj, "n_features_in_") and scaler_obj.n_features_in_ == X_base.shape[1]:
                X_base = scaler_obj.transform(X_base)
            elif hasattr(scaler_obj, "n_features_in_") and scaler_obj.n_features_in_ == 1:
                X_base[:, 0] = scaler_obj.transform(X_base[:, :1]).flatten()
        except Exception:
            pass

    n_echantillons, n_feats = X_base.shape
    if n_feats == nb_attendues:
        return X_base
    elif nb_attendues > n_feats:
        X_adapte = np.zeros((n_echantillons, nb_attendues))
        X_adapte[:, :n_feats] = X_base
        for i in range(n_feats, nb_attendues):
            decalage = i - n_feats + 1
            X_adapte[decalage:, i] = X_base[:-decalage, 0]
            X_adapte[:decalage, i] = X_base[0, 0]
        return X_adapte
    else:
        return X_base[:, :nb_attendues]


modeles, scaler = charger_ressources()

if "donnees" not in st.session_state:
    st.session_state.donnees = None
if "nom_fichier" not in st.session_state:
    st.session_state.nom_fichier = None

# ── BARRE LATÉRALE ──
with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 10px 0;'><div style='font-size:3rem;'>☀️</div></div>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Menu Principal")
    page = st.radio(
        "Sélectionnez une page :", 
        [
            "🏠 Accueil & Présentation", 
            "📂 Importation des Données", 
            "📊 Évaluation & Graphiques",
            "🔮 Prédiction Future"
        ]
    )
    st.divider()
    st.markdown("### 🤖 Configuration IA")
    nom_modele_selectionne = st.selectbox("Modèle d'IA actif :", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele_selectionne]
    nom_court = nom_modele_selectionne.split("—")[0].strip()


# ── PAGE 1 : ACCUEIL & PRÉSENTATION ──
if page == "🏠 Accueil & Présentation":
    if os.path.exists("panneau_pv.jpg"):
        st.image("panneau_pv.jpg", caption="Dispositif expérimental d'acquisition et de production photovoltaïque", use_container_width=True)
    
    st.markdown(
        """
        <div style="text-align: center; margin-top: 25px; margin-bottom: 25px;">
            <span class="badge-pfe">PROJET DE FIN D'ÉTUDES (PFE)</span>
            <h1 style="font-size: 2.3rem;">Prédiction de la Production d'Énergie Photovoltaïque par Intelligence Artificielle</h1>
        </div>
        <hr style="border-color: #DEE2E6; margin-bottom: 30px;"/>
        """, 
        unsafe_allow_html=True
    )
    
    col_gauche, col_droite = st.columns([1, 1], gap="large")
    with col_gauche:
        st.markdown("### 📝 Fiche Technique du Projet")
        st.markdown(
            f"""
            <div class="cadre-accueil">
                <h4 style="color:#E65100; margin-top:0; font-size:1.15rem;">👥 Réalisé par :</h4>
                <ul style="font-size: 1.05rem; line-height: 1.7; margin-bottom: 15px;">
                    <li><b>Nor El Houda BEKADA BENCHAIB</b></li>
                    <li><b>Yousra Oum El kheir HAMMADI</b></li>
                </ul>
                <p style="font-size: 0.9rem; color:#6C757D; margin-bottom:15px;">
                    <i>Spécialité : Automatique / Ingénierie des Systèmes — ESSA Tlemcen</i>
                </p>
                <hr style="margin: 15px 0; border-color: #DEE2E6;"/>
                <h4 style="color:#E65100; font-size:1.15rem;">👨‍🏫 Encadré par :</h4>
                <p style="font-size: 1.05rem; margin-bottom: 5px;"><b>M. Anisse CHIALI</b></p>
                <p style="font-size: 1.05rem;"><b>Mme Imane NEDJAR</b></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with col_droite:
        st.markdown("### 💡 Objectifs & Méthodologie")
        st.markdown(
            """
            Cette plateforme logicielle permet d'automatiser le traitement du signal et l'analyse de vos capteurs. 
            Elle offre une interface interactive pour évaluer et comparer la précision de **5 modèles mathématiques et d'IA** :
            
            * **Approche Linéaire :** Modèle autorégressif externe (ARX).
            * **Réseaux de Neurones :** Perceptron Multicouche (PMC).
            * **Deep Learning Temporel :** Modèles récurrents GRU et LSTM.
            * **Approche Hybride :** Système Neuro-Flou Adaptatif (ANFIS).
            
            L'évaluation s'appuie sur le calcul en temps réel des critères d'erreur ($RMSE$, $MAE$) et du coefficient de détermination ($R^2$).
            """
        )


# ── PAGE 2 : IMPORTATION DES DONNÉES ──
elif page == "📂 Importation des Données":
    st.title("📂 Importation des Données Capteurs")
    fichier_charge = st.file_uploader("Téléverser votre fichier Excel ou CSV provenant des capteurs :", type=["xlsx", "xls", "csv"])
    
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
            st.info(f"ℹ️ Utilisation automatique du fichier local : '{FICHIER_EXEMPLE}'")
        except Exception:
            pass

    if st.session_state.donnees is not None:
        st.markdown(f"### 📋 Tableau de Mesures Actuel (`{st.session_state.nom_fichier}`)")
        st.dataframe(st.session_state.donnees.head(15), use_container_width=True)


# ── PAGE 3 : ÉVALUATION ET GRAPHES ──
elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Traitement & Évaluation des Modèles d'IA")
    if st.session_state.donnees is None:
        st.warning("⚠️ Veuillez d'abord importer un fichier de données dans l'onglet 'Importation des Données'.")
        st.stop()
        
    df = st.session_state.donnees.copy()
    colonnes_manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
    if colonnes_manquantes:
        st.error(f"❌ Colonnes météo manquantes dans votre fichier : {colonnes_manquantes}")
        st.stop()
        
    if COLONNE_CIBLE not in df.columns:
        st.error(f"❌ La colonne cible de puissance mesurée `{COLONNE_CIBLE}` est introuvable.")
        st.stop()

    obj_modele = modeles.get(cle_modele)
    if obj_modele is None:
        st.error(f"❌ Le modèle pour {nom_court} est introuvable. Vérifiez l'emplacement de vos fichiers .pkl/.h5.")
        st.stop()

    if hasattr(obj_modele, "n_features_in_"):
        n_attendues = obj_modele.n_features_in_
    elif hasattr(obj_modele, "input_shape") and obj_modele.input_shape is not None:
        n_attendues = obj_modele.input_shape[-1]
    elif cle_modele == "anfis":
        n_attendues = 3
    else:
        n_attendues = 5

    X_final = preparer_matrice_entrees(df, scaler, cle_modele, n_attendues)
    
    try:
        if cle_modele in ["gru", "lstm"]:
            X_3d = np.reshape(X_final, (X_final.shape[0], 1, X_final.shape[1]))
            y_pred = obj_modele.predict(X_3d, verbose=0).flatten()
        else:
            y_pred = obj_modele.predict(X_final).flatten()
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul mathématique : {e}")
        st.stop()

    y_reel = df[COLONNE_CIBLE].values.astype(float)
    taille_min = min(len(y_reel), len(y_pred))
    y_reel, y_pred = y_reel[:taille_min], y_pred[:taille_min]

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
    mae  = mean_absolute_error(y_reel, y_pred)
    r2   = r2_score(y_reel, y_pred)

    st.markdown(f"### 📐 Indicateurs de Performance — Modèle {nom_court}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="carte-metrique"><div class="valeur">{rmse:.2f}</div><div class="label">RMSE (mW)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="carte-metrique"><div class="valeur">{mae:.2f}</div><div class="label">MAE (mW)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="carte-metrique"><div class="valeur">{r2:.4f}</div><div class="label">R² (Score Global)</div></div>', unsafe_allow_html=True)

    st.markdown("### 📈 Courbes Comparatives")
    indices = list(range(len(y_reel)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=y_reel, name="⚡ Puissance Réelle (Mesurée)", mode="lines", line=dict(color="#1A73E8", width=2)))
    fig.add_trace(go.Scatter(x=indices, y=y_pred, name=f"🤖 Puissance Prédite ({nom_court})", mode="lines", line=dict(color="#FF6B2B", width=2, dash="dash")))
    fig.update_layout(
        title=f"Validation croisée : Puissance réelle mesurée vs Puissance prédite par {nom_court}",
        xaxis=dict(title="Points de mesure (Séquence Temporelle)", gridcolor="#E5E5E5"),
        yaxis=dict(title="Puissance Électrique (mW)", gridcolor="#E5E5E5"),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#F8F9FA", hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


# ── NOUVELLE PAGE 3 : PRÉDICTION FUTURE INTERACTIVE ──
elif page == "🔮 Prédiction Future":
    st.title("🔮 Espace de Test & Prédictions Futures")
    
    obj_modele = modeles.get(cle_modele)
    if obj_modele is None:
        st.error(f"❌ Veuillez d'abord charger le modèle {nom_court} depuis votre espace de stockage.")
        st.stop()

    if hasattr(obj_modele, "n_features_in_"):
        n_attendues = obj_modele.n_features_in_
    elif hasattr(obj_modele, "input_shape") and obj_modele.input_shape is not None:
        n_attendues = obj_modele.input_shape[-1]
    elif cle_modele == "anfis":
        n_attendues = 3
    else:
        n_attendues = 5

    mode_test = st.radio("Choisissez la méthode de test :", ["🎛️ Saisie manuelle par curseurs (Point unique)", "📈 Simulation sur horizon temporel futur"])
    
    if mode_test == "🎛️ Saisie manuelle par curseurs (Point unique)":
        st.markdown("### Ajustez les conditions météo simulées :")
        c1, c2, c3 = st.columns(3)
        with c1:
            val_ldr = h = st.slider("Éclairage Capteur (LDR_Raw)", min_value=0, max_value=4095, value=1400, step=1)
        with c2:
            val_hum = st.slider("Humidité relative (Hum_%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
        with c3:
            val_temp = st.slider("Température Ambiante (Temp_C)", min_value=-5.0, max_value=55.0, value=25.0, step=0.1)
            
        # Création d'une ligne temporaire
        df_temp = pd.DataFrame([{ "LDR_Raw": val_ldr, "Hum_%": val_hum, "Temp_C": val_temp }])
        X_pt = preparer_matrice_entrees(df_temp, scaler, cle_modele, n_attendues)
        
        try:
            if cle_modele in ["gru", "lstm"]:
                X_pt_3d = np.reshape(X_pt, (X_pt.shape[0], 1, X_pt.shape[1]))
                pred_val = obj_modele.predict(X_pt_3d, verbose=0).flatten()[0]
            else:
                pred_val = obj_modele.predict(X_pt).flatten()[0]
            pred_val = max(0.0, pred_val)
            
            st.markdown("---")
            st.markdown(f"#### ⚡ Puissance Électrique Estimée par {nom_court} :")
            st.markdown(f'<div style="background-color:#FFF3CD; border-left:6px solid #FF6B2B; padding:20px; border-radius:8px; font-size:24px; font-weight:bold; color:#E65100;">{pred_val:.2f} mW</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur d'inférence : {e}")
            
    else:
        st.markdown("### Simulation sur un Horizon Temporel")
        if st.session_state.donnees is None:
            st.warning("⚠️ Veuillez importer un fichier de données dans l'onglet 'Importation' pour projeter un horizon.")
            st.stop()
            
        horizon = st.slider("Nombre de points futurs à simuler :", min_value=5, max_value=100, value=30)
        df_horizon = st.session_state.donnees.copy().head(horizon)
        
        X_hor = preparer_matrice_entrees(df_horizon, scaler, cle_modele, n_attendues)
        
        try:
            if cle_modele in ["gru", "lstm"]:
                X_hor_3d = np.reshape(X_hor, (X_hor.shape[0], 1, X_hor.shape[1]))
                preds_hor = obj_modele.predict(X_hor_3d, verbose=0).flatten()
            else:
                preds_hor = obj_modele.predict(X_hor).flatten()
            preds_hor = np.clip(preds_hor, a_min=0, a_max=None)
            
            fig_futur = go.Figure()
            fig_futur.add_trace(go.Scatter(y=preds_hor, mode="lines+markers", name="Prédiction Horizon", line=dict(color="#E65100", width=3)))
            fig_futur.update_layout(
                title=f"Horizon Prévisionnel de Puissance Électrique ({nom_court})",
                xaxis=dict(title="Index Temporel Futur (+t)", gridcolor="#E5E5E5"),
                yaxis=dict(title="Puissance Prédite (mW)", gridcolor="#E5E5E5"),
                plot_bgcolor="#FFFFFF", paper_bgcolor="#F8F9FA"
            )
            st.plotly_chart(fig_futur, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur calcul horizon : {e}")
