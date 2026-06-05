# ==========================================
# BARRE LATÉRALE (Navigation)
# ==========================================
st.sidebar.title(" Navigation")
section = st.sidebar.radio(
    "Aller vers :",
    [" 1. Accueil", " 2. Données utilisateur", " 3. Modèles IA", " 4. Prédiction"]
)

# ==========================================
# LOGIQUE DE NAVIGATION
# ==========================================

if section == " 1. Accueil":
    st.title(" Prototype de Plateforme d'IA pour la Prévision PV")
    # ... (votre code accueil)

elif section == " 2. Données utilisateur":
    st.title(" Données Utilisateur & Station Locale")
    # ... (votre code données)

elif section == " 3. Modèles IA":
    st.title(" Évaluation des Modèles d'Intelligence Artificielle")
    # ... (votre code modèles)

elif section == " 4. Prédiction":
    st.title(" Prévision de la Puissance Photovoltaïque")
    # --- INSÉREZ ICI LE CODE DE PRÉDICTION CORRIGÉ QUE JE VOUS AI DONNÉ AVANT ---
    if st.session_state["df_user"] is None:
        st.error("Veuillez charger vos données.")
    else:
        # Votre logique de prédiction ici...
        pass
