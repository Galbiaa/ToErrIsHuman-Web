import sys
from pathlib import Path
import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

# Configurazione pagina Streamlit
st.set_page_config(
    page_title="ToErrIsHuman AI",
    page_icon="🧠",
    layout="wide"
)

# Includi la cartella src nel percorso Python
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import DiagnosticNet

device = torch.device("cpu")

# Trasformazioni per l'inferenza della ResNet-18 (Stage 1)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def logits_to_prob(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))

# Caricamento e caching dei 5 fold in RAM per massima velocità
@st.cache_resource
def load_ensemble_models():
    diagnostic_models = []
    preprocessors = []
    xgb_models = []
    thresholds = []

    for fold in range(1, 6):
        diag_path = Path(f"models/diagnostic/diagnostic_fold_{fold}.pt")
        pre_path = Path(f"models/D/preprocessor_fold_{fold}.joblib")
        d_path = Path(f"models/D/scenario_D_fold_{fold}.joblib")

        # 1. Carica ResNet-18 (Diagnostic Model)
        ckpt = torch.load(diag_path, map_location=device, weights_only=False)
        arch = ckpt.get("architecture", {})
        model = DiagnosticNet(
            pretrained=False,
            freeze_backbone=False,
            aggregation=arch.get("aggregation", "concat"),
            hidden_dim=int(arch.get("hidden_dim", 256)),
            dropout=float(ckpt.get("dropout", 0.35)),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        diagnostic_models.append(model)

        # 2. Carica Preprocessor, XGBoost e la soglia di Youden esatta
        pre = joblib.load(pre_path)
        payload = joblib.load(d_path)
        clf = payload["model"]
        
        # Lettura esatta della chiave 'threshold_youden' salvata nei .joblib
        thr = float(payload.get("threshold_youden", 0.20))

        preprocessors.append(pre)
        xgb_models.append(clf)
        thresholds.append(thr)

    return diagnostic_models, preprocessors, xgb_models, thresholds

def predict_single_fold(model, axial_img, coronal_img, sagittal_img):
    t_ax = transform(axial_img)
    t_co = transform(coronal_img)
    t_sa = transform(sagittal_img)
    images_tensor = torch.stack([t_ax, t_co, t_sa], dim=0).unsqueeze(0)
    with torch.no_grad():
        logits = model(images_tensor)
        prob = float(logits_to_prob(logits.numpy())[0])
    return prob

# Interfaccia grafica Streamlit
st.title("🧠 ToErrIsHuman — Predictor di Errore Diagnostico Umano")
st.markdown("""
Questa applicazione web stima la probabilità di **errore diagnostico umano** combinando l'analisi radiologica 3D delle MRI con il contesto clinico del medico valutatore (Rater).
""")

# Carica i modelli in RAM
with st.spinner("Caricamento modelli in memoria..."):
    diagnostic_models, preprocessors, xgb_models, thresholds = load_ensemble_models()

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 1. Carica le 3 Viste MRI del Caso")
    ax_file = st.file_uploader("Vista Assiale (Axial)", type=["jpg", "jpeg", "png"])
    co_file = st.file_uploader("Vista Coronale (Coronal)", type=["jpg", "jpeg", "png"])
    sa_file = st.file_uploader("Vista Sagittale (Sagittal)", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("👨‍⚕️ 2. Dati del Medico Evaluator")
    rating = st.radio(
        "Diagnosi del Medico (Rating)", 
        options=[0, 1], 
        format_func=lambda x: "0 - Esame Negativo / Sano" if x == 0 else "1 - Esame Positivo / Patologico"
    )
    confidence = st.slider("Confidenza del Medico (1 = Minima, 5 = Massima)", 1, 5, 4)
    difficulty = st.slider("Difficoltà Percepita del Caso (1-5)", 1, 5, 3)
    expertise = st.number_input("Esperienza del Medico (Anni di attività)", min_value=0, max_value=50, value=5)

st.divider()

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

            for fold in range(5):
                model = diagnostic_models[fold]
                pre = preprocessors[fold]
                clf = xgb_models[fold]

                # 1. Probabilità Patologia AI (Stadio 1)
                p_ai = predict_single_fold(model, img_ax, img_co, img_sa)
                ai_probs.append(p_ai)

                # 2. Calcolo Feature Derivate per Modello D (Stadio 2)
                ai_pred_class = 1 if p_ai >= 0.5 else 0
                disagreement = 1 if rating != ai_pred_class else 0
                p_wrong = p_ai if rating == 0 else (1.0 - p_ai)
                margin = abs(p_ai - 0.5)

                X_df = pd.DataFrame([{
                    "rating-confidence": float(confidence),
                    "case-difficulty": float(difficulty),
                    "rater-expertise": float(expertise),
                    "rater-accuracy": 0.80,  # Fallback per medici non presenti nel dataset storico
                    "rater-confidence": float(confidence),
                    "rating": int(rating),
                    "ai_probability_class_1": float(p_ai),
                    "ai_predicted_class": int(ai_pred_class),
                    "rater_ai_disagreement": int(disagreement),
                    "ai_probability_rater_wrong": float(p_wrong),
                    "ai_margin": float(margin)
                }])

                # 3. Probabilità Errore Diagnostico con XGBoost (Stadio 2)
                X_t = pre.transform(X_df)  # Passa il DataFrame per preservare i nomi delle colonne
                p_err = float(clf.predict_proba(X_t)[:, 1][0])
                d_probs.append(p_err)

            mean_p_error = float(np.mean(d_probs))
            mean_p_ai = float(np.mean(ai_probs))
            mean_threshold = float(np.mean(thresholds))

        st.subheader("📊 Risultati dell'Inferenza")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(label="Stima Rischio Errore Umano", value=f"{mean_p_error:.1%}")
        m_col2.metric(label="Soglia Operativa Youden (Ensemble)", value=f"{mean_threshold:.1%}")
        m_col3.metric(label="Probabilità Patologia (AI Stadio 1)", value=f"{mean_p_ai:.1%}")

        st.divider()

        # Logica di valutazione del rischio rispetto alla soglia Youden reale
        if mean_p_error >= mean_threshold:
            st.error(
                f"🚨 **ALTO RISCHIO ERRORE DIAGNOSTICO**\n\n"
                f"La probabilità stimata di errore umano (**{mean_p_error:.1%}**) supera o eguaglia la soglia operativa del sistema (**{mean_threshold:.1%}**).\n\n"
                f"👉 *Raccomandazione:* Si consiglia un secondo parere o una revisione approfondita dell'esame radiologico."
            )
        else:
            st.success(
                f"✅ **BASSO RISCHIO ERRORE**\n\n"
                f"La probabilità stimata di errore umano (**{mean_p_error:.1%}**) è inferiore alla soglia operativa (**{mean_threshold:.1%}**).\n\n"
                f"👉 *Raccomandazione:* La diagnosi del medico appare coerente con il profilo clinico ed i rilievi delle immagini MRI."
            )
