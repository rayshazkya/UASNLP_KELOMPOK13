"""
utils.py
Fungsi-fungsi pembantu untuk dashboard HoaxBERT:
- load model & tokenizer (IndoBERT / RoBERTa) dari saved_models/
- fungsi prediksi teks
"""

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path relatif dari app/app.py ke folder model.
# Sesuaikan kalau struktur folder kamu berbeda.
MODEL_PATHS = {
    "IndoBERT": "../content/saved_models/indobert",
    "RoBERTa": "../content/saved_models/roberta",
}


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """
    Load tokenizer & model untuk satu kali saja, lalu di-cache oleh Streamlit
    supaya tidak reload tiap kali user klik tombol prediksi.
    """
    path = MODEL_PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


def predict(text: str, model_name: str, max_length: int = 256) -> dict:
    """
    Prediksi label hoaks/non-hoaks untuk satu teks menggunakan model tertentu.

    Return:
        {
            "label": "hoaks" / "non-hoaks",
            "confidence": float (0-1),
            "probs": {"non-hoaks": float, "hoaks": float}
        }
    """
    tokenizer, model = load_model(model_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()

    id2label = model.config.id2label
    pred_id = int(torch.argmax(logits, dim=-1).item())

    prob_dict = {id2label[i]: probs[i] for i in range(len(probs))}

    return {
        "label": id2label[pred_id],
        "confidence": probs[pred_id],
        "probs": prob_dict,
    }


def load_eval_metrics() -> dict:
    """
    Placeholder metrik evaluasi hasil training.
    GANTI angka di bawah ini dengan hasil evaluasi asli dari notebook training kamu
    (accuracy, precision, recall, f1, roc_auc untuk masing-masing model).
    """
    return {
        "IndoBERT": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
        },
        "RoBERTa": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
        },
    }
