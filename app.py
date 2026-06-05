# --- SECTION 4 : PRÉDICTION (Version Corrigée & Robuste) ---
elif section == " 4. Prédiction":
    st.title(" Prévision de la Puissance Photovoltaïque")
    
    if st.session_state["df_user"] is None:
        st.error("Veuillez charger vos données.")
    else:
        # 1. Sélection des entrées (Sliders)
        col_m, col_h = st.columns(2)
        with col_m: modele_pred = st.selectbox("Modèle :", list(dict_modeles.keys()))
        
        # Sliders pour les inputs (LDR, Hum, Temp)
        ldr = st.slider("Éclairement (LDR_Raw)", 0, 2000, 500)
        hum = st.slider("Humidité (Hum_%)", 0, 100, 50)
        tmp = st.slider("Température (Temp_C)", -10, 50, 25)
        
        # 2. Préparation sécurisée des données
        input_data = pd.DataFrame([[ldr, hum, tmp]], columns=["LDR_Raw", "Hum_%", "Temp_C"])
        
        # Application du scaler pour éviter les R² négatifs
        try:
            input_scaled = scaler_global.transform(input_data)
        except:
            input_scaled = input_data.values # Fallback si problème de scaler

        # 3. Adaptation dynamique aux modèles
        model = dict_modeles[modele_pred]
        
        try:
            if "LSTM" in modele_pred or "GRU" in modele_pred:
                # Format 3D : (1, 1, 3)
                x_final = input_scaled.reshape((1, 1, 3))
            else:
                # Format 2D : (1, 3)
                x_final = input_scaled
            
            # 4. Prédiction avec sécurité
            prediction = model.predict(x_final)
            
            # Extraction propre de la valeur
            valeur_predite = float(np.array(prediction).flatten()[0])
            
            # Correction des valeurs aberrantes (force le positif)
            valeur_predite = max(0.0, valeur_predite)
            
            # Affichage résultat
            st.metric(label="Puissance Prévue (mW)", value=f"{valeur_predite:.2f} mW")
            
        except Exception as e:
            st.error(f"Erreur de dimension sur le modèle : {e}")
