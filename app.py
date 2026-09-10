def predict_single_fold(model, axial_img, coronal_img, sagittal_img):
    t_ax = transform(axial_img)
    t_co = transform(coronal_img)
    t_sa = transform(sagittal_img)
    images_tensor = torch.stack([t_ax, t_co, t_sa], dim=0).unsqueeze(0)
    with torch.no_grad():
        logits = model(images_tensor)
        prob = float(logits_to_prob(logits.numpy())[0])
    return prob

# Nel pulsante "Calcola Rischio Errore":
if st.button("🔍 Calcola Rischio Errore", type="primary", use_container_width=True):
    if not (ax_file and co_file and sa_file):
        st.error("⚠️ Carica tutte e tre le immagini MRI per procedere!")
    else:
        with st.spinner("Elaborazione inferenza Ensemble (5 Fold)..."):
            img_ax = Image.open(ax_file).convert("RGB")
            img_co = Image.open(co_file).convert("RGB")
            img_sa = Image.open(sa_file).convert("RGB")

            d_probs = []
            ai_probs = []
            thresholds = []

            for fold in range(5):
                model = diagnostic_models[fold]
                pre = preprocessors[fold]
                clf = xgb_models[fold]

                # 1. AI Probability (Stadio 1)
                p_ai = predict_single_fold(model, img_ax, img_co, img_sa)
                ai_probs.append(p_ai)

                # 2. Prepara Feature
                ai_pred_class = 1 if p_ai >= 0.5 else 0
                disagreement = 1 if rating != ai_pred_class else 0
                p_wrong = p_ai if rating == 0 else (1.0 - p_ai)
                margin = abs(p_ai - 0.5)

                X_df = pd.DataFrame([{
                    "rating-confidence": float(confidence),
                    "case-difficulty": float(difficulty),
                    "rater-expertise": float(expertise),
                    "rater-accuracy": 0.80, # valore storico medio
                    "rater-confidence": float(confidence),
                    "rating": int(rating),
                    "ai_probability_class_1": p_ai,
                    "ai_predicted_class": ai_pred_class,
                    "rater_ai_disagreement": disagreement,
                    "ai_probability_rater_wrong": p_wrong,
                    "ai_margin": margin
                }])

                # 3. XGBoost Probability (Stadio 2) e Soglia Youden
                payload = joblib.load(Path(f"models/D/scenario_D_fold_{fold+1}.joblib"))
                thr = float(payload.get("threshold_youden", 0.5))
                thresholds.append(thr)

                X_t = pre.transform(X_df.values)
                p_err = float(clf.predict_proba(X_t)[:, 1][0])
                d_probs.append(p_err)

            mean_p_error = float(np.mean(d_probs))
            mean_p_ai = float(np.mean(ai_probs))
            mean_threshold = float(np.mean(thresholds))

        st.subheader("📊 Risultati dell'Inferenza")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(label="Stima Rischio Errore Umano", value=f"{mean_p_error:.1%}")
        m_col2.metric(label="Soglia Operativa del Modello (Youden)", value=f"{mean_threshold:.1%}")
        m_col3.metric(label="Probabilità Patologia stimata dall'AI", value=f"{mean_p_ai:.1%}")

        # Confronto rigoroso con la vera soglia del modello
        if mean_p_error >= mean_threshold:
            st.error(f"🚨 **ALTO RISCHIO ERRORE DIAGNOSTICO** (Probabilità {mean_p_error:.1%} ≥ Soglia {mean_threshold:.1%})\n\nL'AI indica una probabilità elevata che la diagnosi del rater contenga un errore. Si consiglia un secondo parere radiologico.")
        else:
            st.success(f"✅ **BASSO RISCHIO ERRORE** (Probabilità {mean_p_error:.1%} < Soglia {mean_threshold:.1%})\n\nLa valutazione del medico appare coerente con le caratteristiche radiologiche ed il contesto clinico.")
