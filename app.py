"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PLATEFORME DE PRÉDICTION D'ÉNERGIE PHOTOVOLTAÏQUE PAR IA            ║
║                Projet de Fin d'Études — Ingénierie des Systèmes              ║
╚══════════════════════════════════════════════════════════════════════════════╝
Fichier  : app.py
Auteurs  : Nor El Houda BEKADA BENCHAIB & Yousra Oum El kheir HAMMADI
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

import plotly.graph_objects as go

# 1. Configuration de la page (Doit être la toute première commande Streamlit)
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

# 3. Chargement robuste des fichiers d'IA et du Scaler
@st.cache_resource(show_spinner="⚙️ Chargement des architectures d'IA…")
def charger_ressources():
    modeles = {}
    scaler = None

    # Chargement du scaler
    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
        except Exception as e:
            st.warning(f"ℹ️ Info Scaler : {e}")

    # Modèles Scikit-Learn / Joblib
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

    # Modèles Deep Learning Keras/TensorFlow
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
    except ImportError:
        pass

    return modeles, scaler

# 4. Sécurisation de la matrice pour éviter les erreurs de dimensions (features)
def preparer_matrice_entrees(df_data: pd.DataFrame, scaler_obj, cle_mod, nb_attendues) -> np.ndarray:
    X_base = df_data[COLONNES_REQUISES].values.astype(float)
    if np.any(np.isnan(X_base)):
        X_base = np.nan_to_num(X_base, nan=0.0)

    # Normalisation adaptative sécurisée
    if scaler_obj is not None:
        try:
            if hasattr(scaler_obj, "n_features_in_") and scaler_obj.n_features_in_ == X_base.shape[1]:
                X_base = scaler_obj.transform(X_base)
            elif hasattr(scaler_obj, "n_features_in_") and scaler_obj.n_features_in_ == 1:
                # Si le scaler n'attend qu'une variable (ex: l'éclairage seul)
                X_base[:, 0] = scaler_obj.transform(X_base[:, :1]).flatten()
        except Exception:
            pass

    n_echantillons, n_feats = X_base.shape
    if n_feats == nb_attendues:
        return X_base
    elif nb_attendues > n_feats:
        # Reconstruction dynamique des décalages temporels (ex: ARX à 5 entrées)
        X_adapte = np.zeros((n_echantillons, nb_attendues))
        X_adapte[:, :n_feats] = X_base
        for i in range(n_feats, nb_attendues):
            decalage = i - n_feats + 1
            X_adapte[decalage:, i] = X_base[:-decalage, 0]
            X_adapte[:decalage, i] = X_base[0, 0]
        return X_adapte
    else:
        return X_base[:, :nb_attendues]


# Initialisation et chargement des variables globales
modeles, scaler = charger_ressources()

if "donnees" not in st.session_state:
    st.session_state.donnees = None
if "nom_fichier" not in st.session_state:
    st.session_state.nom_fichier = None

# ── BARRE LATÉRALE (NAVIGATION) ──
with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 10px 0;'><div style='font-size:3rem;'>☀️</div></div>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Menu Principal")
    page = st.radio(
        "Sélectionnez une page :", 
        [
            "🏠 Accueil & Présentation", 
            "📂 Importation des Données", 
            "📊 Évaluation & Graphiques"
        ]
    )
    st.divider()
    st.markdown("### 🤖 Configuration IA")
    nom_modele_selectionne = st.selectbox("Modèle d'IA actif :", list(MODELES_DISPONIBLES.keys()))
    cle_modele = MODELES_DISPONIBLES[nom_modele_selectionne]
    nom_court = nom_modele_selectionne.split("—")[0].strip()


# ── 1. NOUVELLE PAGE D'ACCUEIL AVEC PHOTO EN BANNIÈRE LARGE ──
if page == "🏠 Accueil & Présentation":
    
    # Affichage de l'image sur toute la largeur (use_container_width=True) au tout début
    if os.path.exists("panneau_pv.jpg"):
        st.image(
            "panneau_pv.jpg", 
            caption="Dispositif expérimental d'acquisition et de production photovoltaïque", 
            use_container_width=True
        )
    else:
        st.info("💡 Pour afficher votre photo ici sur toute la largeur, téléversez 'panneau_pv.jpg' sur GitHub.")

    # Titre centralisé sous l'image
    st.markdown(
        """
        <div style="text-align: center; margin-top: 25px; margin-bottom: 25px;">
            <span class="badge-pfe">PROJET DE FIN D'ÉTUDES (PFE)</span>
            <h1 style="font-size: 2.3rem;">Prédiction de la Production d'Énergie Photovoltaïque par Intelligence Artificielle</h1>
            <p style="font-size: 1.1rem; color: #495057; max-width: 900px; margin: 0 auto; padding-top: 8px;">
                Application d'ingénierie des systèmes et de traitement du signal dédiée à la modélisation prédictive 
                et à l'analyse comparative des performances de modèles d'IA à partir de données météorologiques.
            </p>
        </div>
        <hr style="border-color: #DEE2E6; margin-bottom: 30px;"/>
        """, 
        unsafe_allow_html=True
    )
    
    # Organisation des informations en colonnes sous la bannière
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
                <p style="font-size: 0.9rem; color:#6C757D; margin-bottom:15px;">
                    <i>Spécialité : Automatique / Ingénierie des Systèmes — ESSA Tlemcen</i>
                </p>
                <hr style="margin: 15px 0; border-color: #DEE2E6;"/>
                <h4 style="color:#E65100; font-size:1.15rem;">👨‍🏫 Encadré par :</h4>
                <p style="font-size: 1.05rem; margin-bottom: 5px;"><b>Pr. Mohammed Sahlaoui</b></p>
                <p style="font-size: 1.05rem;"><b>Dr. Abdelkader Ghezouani</b></p>
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


# ── 2. PAGE : IMPORTATION DES DONNÉES ──
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
        st.markdown(f"### 📋 Tableau de Mesures Actuel (`{st.session_state.nom_fichier}`)" )
        st.dataframe(st.session_state.donnees.head(15), use_container_width=True)


# ── 3. PAGE : ÉVALUATION ET GRAPHES ──
elif page == "📊 Évaluation & Graphiques":
    st.title("📊 Traitement & Évaluation des Modèles d'IA")
    
    if st.session_state.donnees is None:
        st.warning("⚠️ Veuillez d'abord importer ou charger un fichier de données dans l'onglet 'Importation des Données'.")
        st.stop()
        
    df = st.session_state.donnees.copy()
    
    # Validation stricte des colonnes réelles du fichier Excel
    colonnes_manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
    if colonnes_manquantes:
        st.error(f"❌ Colonnes météo manquantes dans votre fichier : {colonnes_manquantes}")
        st.info("Votre fichier Excel doit obligatoirement contenir les en-têtes exacts suivants : `LDR_Raw`, `Hum_%` et `Temp_C`")
        st.stop()
        
    if COLONNE_CIBLE not in df.columns:
        st.error(f"❌ La colonne cible de puissance mesurée `{COLONNE_CIBLE}` est introuvable dans le fichier.")
        st.stop()

    obj_modele = modeles.get(cle_modele)
    if obj_modele is None:
        st.error(f"❌ Le modèle binaire pour {nom_court} est introuvable ou n'a pas pu être chargé depuis votre dépôt.")
        st.stop()

    # Détection automatique du nombre d'entrées attendues par l'architecture
    if hasattr(obj_modele, "n_features_in_"):
        n_attendues = obj_modele.n_features_in_
    elif hasattr(obj_modele, "input_shape") and obj_modele.input_shape is not None:
        n_attendues = obj_modele.input_shape[-1]
    elif cle_modele == "anfis":
        n_attendues = 3
    else:
        n_attendues = 5

    # Préparation et normalisation
    X_final = preparer_matrice_entrees(df, scaler, cle_modele, n_attendues)
    
    # Exécution des prédictions selon la nature du modèle
    try:
        if cle_modele in ["gru", "lstm"]:
            X_3d = np.reshape(X_final, (X_final.shape[0], 1, X_final.shape[1]))
            y_pred = obj_modele.predict(X_3d, verbose=0).flatten()
        else:
            y_pred = obj_modele.predict(X_final).flatten()
            
        # Sécurité : Pas de puissance négative possible physiquement
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul mathématique du modèle {nom_court} : {e}")
        st.stop()

    y_reel = df[COLONNE_CIBLE].values.astype(float)
    taille_min = min(len(y_reel), len(y_pred))
    y_reel, y_pred = y_reel[:taille_min], y_pred[:taille_min]

    # Calcul des indicateurs de performance (Traitement du signal)
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

    # Graphique interactif Plotly
    st.markdown("### 📈 Courbes Comparatives")
    indices = list(range(len(y_reel)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices, y=y_reel,
        name="⚡ Puissance Réelle (Mesurée)",
        mode="lines", line=dict(color="#1A73E8", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=y_pred,
        name=f"🤖 Puissance Prédite ({nom_court})",
        mode="lines", line=dict(color="#FF6B2B", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title=f"Validation croisée : Puissance réelle mesurée vs Puissance prédite par {nom_court}",
        xaxis=dict(title="Points de mesure (Séquence Temporelle)", gridcolor="#E5E5E5"),
        yaxis=dict(title="Puissance Électrique (mW)", gridcolor="#E5E5E5"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F8F9FA",
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
