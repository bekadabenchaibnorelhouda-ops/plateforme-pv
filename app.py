import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(page_title="Plateforme IA - Photovoltaïque", page_icon="☀️", layout="wide")

# ==========================================
# CHARGEMENT DES RESSOURCES TECHNIQUES
# ==========================================
@st.cache_resource
def charger_les_modeles():
    modeles = {}
    scaler = None
    
    # Chargement du scaler s'il existe
    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
        except:
            pass

    # Modèles Scikit-Learn / Joblib
    fichiers_pkl = {
        "MLP (Multi-Layer Perceptron)": "model_mlp.pkl",
        "ARX": "model_arx.pkl",
        "ANFIS (Random Forest)": "model_anfis.pkl"
    }
    for nom, fichier in fichiers_pkl.items():
        if os.path.exists(fichier):
            try:
                modeles[nom] = joblib.load(fichier)
            except:
                pass

    # Modèles Deep Learning (Keras/TensorFlow)
    try:
        from tensorflow.keras.models import load_model
        fichiers_h5 = {
            "LSTM (Long Short-Term Memory)": "model_lstm.h5",
            "GRU (Gated Recurrent Unit)": "model_gru.h5"
        }
        for nom, fichier in fichiers_h5.items():
            if os.path.exists(fichier):
                try:
                    modeles[nom] = load_model(fichier, compile=False)
                except:
                    pass
    except ImportError:
        pass

    return modeles, scaler

dict_modeles, scaler_global = charger_les_modeles()

# Initialisation du Session State
if "df_user" not in st.session_state:
    st.session_state["df_user"] = None

COLONNES_REQUISES = ["LDR_Raw", "Hum_%", "Temp_C"]
COLONNE_CIBLE = "Puissance_mW"

# ==========================================
# BARRE LATÉRALE (Navigation)
# ==========================================
st.sidebar.title("☀️ Menu de Contrôle")
section = st.sidebar.radio(
    "Aller vers :",
    ["1. Accueil", "2. Données utilisateur", "3. Évaluation Réelle des Modèles", "4. Simulation Future"]
)

# ==========================================
# SECTION 1 : ACCUEIL
# ==========================================
if section == "1. Accueil":
    st.title(" Prototype de Plateforme d'IA pour la Prévision PV")
    st.subheader("Solution intelligente d'aide à la décision énergétique")
    
    st.markdown("""
    ### Description du Projet
    Cette plateforme basée sur l'**Intelligence Artificielle** permet la prévision à court terme de la production d'un système photovoltaïque à partir de données météorologiques.
    
    **Moteur de calcul actif :** L'application traite désormais vos modèles entraînés en temps réel pour analyser vos jeux de tests.
    """)

# ==========================================
# SECTION 2 : DONNÉES UTILISATEUR
# ==========================================
elif section == "2. Données utilisateur":
    st.title(" Données Utilisateur & Station Locale")
    uploaded_file = st.file_uploader("Choisissez votre fichier (Excel ou CSV)", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df_raw = pd.read_excel(uploaded_file)
            else:
                df_raw = pd.read_csv(uploaded_file)
            
            df_cleaned = df_raw.copy().dropna(subset=COLONNES_REQUISES + [COLONNE_CIBLE])
            
            if "Date" in df_cleaned.columns: df_cleaned["Date"] = df_cleaned["Date"].astype(str)
            if "Heure" in df_cleaned.columns: df_cleaned["Heure"] = df_cleaned["Heure"].astype(str)
                
            st.session_state["df_user"] = df_cleaned
            st.success(" Fichier chargé et synchronisé en mémoire !")
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")
            
    if st.session_state["df_user"] is not None:
        st.dataframe(st.session_state["df_user"].head())
    else:
        st.info(" En attente de l'importation d'un fichier.")

# ==========================================
# SECTION 3 : ÉVALUATION RÉELLE (CORRIGÉE)
# ==========================================
elif section == "3. Évaluation Réelle des Modèles":
    st.title("📊 Évaluation Réelle & Calcul des Métriques")
    
    if st.session_state["df_user"] is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier dans la section 2.")
        st.stop()
        
    df_global = st.session_state["df_user"].copy()
    
    # Application automatique du fractionnement conforme à votre script piéton (70/15/15)
    st.markdown("### ⚙️ Alignement du Dataset (Validation Croisée)")
    choix_split = st.radio("Évaluer le modèle sur la portion :", ["📈 Échantillons de Test (Derniers 15% du fichier)", "📊 Fichier Complet (100%)"])
    
    total_lignes = len(df_global)
    idx_val = int(total_lignes * 0.85)
    
    if choix_split == "📈 Échantillons de Test (Derniers 15% du fichier)":
        df_analyse = df_global.iloc[idx_val:].reset_index(drop=True)
    else:
        df_analyse = df_global.reset_index(drop=True)
        
    modele_choisi = st.selectbox("Sélectionnez le modèle à faire tourner :", list(dict_modeles.keys()))
    obj_modele = dict_modeles[modele_choisi]
    
    # --- INSPECTION DYNAMIQUE DES FEATURES REQUISES ---
    nb_features_attendues = 3
    if "LSTM" in modele_choisi or "GRU" in modele_choisi:
        if hasattr(obj_modele, "input_shape") and obj_modele.input_shape is not None:
            nb_features_attendues = obj_modele.input_shape[-1]
    elif "ARX" in modele_choisi or "ANFIS" in modele_choisi:
        nb_features_attendues = 1  # Ajustement standard pour vos modèles mono-variable
    elif hasattr(obj_modele, "n_features_in_"):
        nb_features_attendues = obj_modele.n_features_in_

    # Extraction des features selon les exigences du modèle choisi
    if nb_features_attendues == 1:
        X = df_analyse[["LDR_Raw"]].values.astype(float)
        if scaler_global is not None:
            try:
                if hasattr(scaler_global, "mean_") and len(scaler_global.mean_) >= 1:
                    X = (X - scaler_global.mean_[0]) / np.sqrt(scaler_global.var_[0])
            except: pass
    else:
        X = df_analyse[COLONNES_REQUISES].values.astype(float)
        if scaler_global is not None:
            try: X = scaler_global.transform(X)
            except: pass

    # Sécurité anti-valeurs manquantes
    X = np.nan_to_num(X, nan=0.0)
    y_reel = df_analyse[COLONNE_CIBLE].values.astype(float)

    # Prédiction réelle
    try:
        if "LSTM" in modele_choisi or "GRU" in modele_choisi:
            X_rnn = np.reshape(X, (X.shape[0], 1, X.shape[1]))
            y_pred = obj_modele.predict(X_rnn, verbose=0).flatten()
        else:
            y_pred = obj_modele.predict(X).flatten()
        
        y_pred = np.clip(y_pred, 0, None)
        
        # Calcul des vrais indicateurs mathématiques
        rmse_val = np.sqrt(mean_squared_error(y_reel, y_pred))
        mae_val  = mean_absolute_error(y_reel, y_pred)
        r2_val   = r2_score(y_reel, y_pred)

        # Affichage dynamique des vrais résultats
        c1, c2, c3 = st.columns(3)
        with c1: st.metric(label="R² Calculé", value=f"{r2_val:.4f}")
        with c2: st.metric(label="RMSE Réel", value=f"{rmse_val:.2f} mW")
        with c3: st.metric(label="MAE Réel", value=f"{mae_val:.2f} mW")
        
        # Tracé de la courbe réelle vs prédiction
        st.subheader("📈 Graphique de Performance Réelle")
        df_graphe = pd.DataFrame({
            "Production Réelle (mW)": y_reel[:150],
            f"Prédiction {modele_choisi} (mW)": y_pred[:150]
        })
        st.line_chart(df_graphe)

    except Exception as e:
        st.error(f"Le modèle n'a pas pu s'exécuter. Erreur de format d'entrée : {e}")

# ==========================================
# SECTION 4 : SIMULATION INTERACTIVE
# ==========================================
elif section == "4. Simulation Future":
    st.title("🔮 Générateur de Prévisions")
    if st.session_state["df_user"] is None:
        st.warning("⚠️ Importez d'abord vos données.")
        st.stop()
        
    modele_pred = st.selectbox("Sélectionnez le modèle pour la simulation :", list(dict_modeles.keys()))
    st.info(f"Le modèle {modele_pred} est actif pour simuler le comportement du panneau.")
