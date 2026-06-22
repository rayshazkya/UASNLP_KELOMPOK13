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


def render_gauge_html(percent: float, color: str, sublabel: str = "Confidence") -> str:
    """
    Bangun HTML untuk gauge melingkar (conic-gradient) yang menampilkan
    persentase confidence di tengahnya.

    PENTING: string dibangun TANPA indentasi/newline berlebih. Markdown
    menganggap baris yang diawali 4+ spasi sebagai blok kode (literal text,
    bukan HTML), jadi semua tag di sini sengaja dirapatkan jadi satu baris.
    """
    pct = round(percent * 100, 1)
    return (
        f'<div class="gauge-wrap">'
        f'<div class="gauge" style="background: conic-gradient({color} {pct}%, #e5e9f2 {pct}% 100%);">'
        f'<div class="gauge-inner">'
        f'<span class="gauge-percent">{pct:.1f}%</span>'
        f'<span class="gauge-label">{sublabel}</span>'
        f'</div></div></div>'
    )


def label_badge_html(label: str) -> str:
    """Badge kecil berwarna untuk label hoaks / non-hoaks."""
    css_class = "badge-hoaks" if label == "hoaks" else "badge-nonhoaks"
    return f'<span class="label-badge {css_class}">{label.upper()}</span>'


def gauge_color(label: str) -> str:
    """Warna gauge sesuai label: merah untuk hoaks, teal untuk non-hoaks."""
    return "#ef4444" if label == "hoaks" else "#14b8a6"


def render_metrics_card_html(model_name: str, metrics: dict) -> str:
    """
    Bangun satu kartu metrik utuh (header + grid metrik) sebagai SATU blok HTML
    rapat (tanpa indentasi), supaya tidak ditafsirkan sebagai code block oleh
    Markdown dan tidak pecah saat dirender Streamlit.
    """
    rows = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1-score", metrics["f1"]),
        ("ROC-AUC", metrics["roc_auc"]),
    ]
    items_html = "".join(
        f'<div class="metric-item">'
        f'<div class="metric-label">{name}</div>'
        f'<div class="metric-value">{value:.2%}</div>'
        f'</div>'
        for name, value in rows
    )
    return (
        f'<div class="card">'
        f'<div class="card-header">{model_name}</div>'
        f'<div class="metric-grid">{items_html}</div>'
        f'</div>'
    )


def render_result_card_html(model_name: str, result: dict) -> str:
    """
    Bangun satu kartu hasil prediksi utuh (icon + header + badge label + gauge)
    sebagai SATU blok HTML rapat (tanpa indentasi).
    """
    label = result["label"]
    badge = label_badge_html(label)
    gauge = render_gauge_html(result["confidence"], gauge_color(label))
    icon = model_icon_html(model_name)
    return (
        f'<div class="card">'
        f'<div class="card-icon-row">{icon}'
        f'<div class="card-header" style="margin:0;">{model_name}</div></div>'
        f'{badge}'
        f'{gauge}'
        f'</div>'
    )


def hero_illustration_svg() -> str:
    """Ilustrasi dekoratif: dokumen + kaca pembesar + badge centang, untuk header."""
    return (
        '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="100" cy="100" r="90" fill="#eafaf7"/>'
        '<rect x="55" y="40" width="75" height="100" rx="10" fill="#ffffff" stroke="#14b8a6" stroke-width="3"/>'
        '<rect x="68" y="58" width="49" height="6" rx="3" fill="#cbd5e1"/>'
        '<rect x="68" y="72" width="49" height="6" rx="3" fill="#cbd5e1"/>'
        '<rect x="68" y="86" width="35" height="6" rx="3" fill="#cbd5e1"/>'
        '<circle cx="128" cy="118" r="26" fill="#ffffff" stroke="#0d9488" stroke-width="4"/>'
        '<line x1="147" y1="137" x2="165" y2="155" stroke="#0d9488" stroke-width="6" stroke-linecap="round"/>'
        '<circle cx="128" cy="118" r="14" fill="#d9f4ef"/>'
        '<circle cx="158" cy="55" r="16" fill="#fb7185"/>'
        '<path d="M151 55 l5 5 l10 -10" stroke="white" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )


def model_icon_html(model_name: str) -> str:
    """Badge inisial berwarna untuk tiap model (I untuk IndoBERT, R untuk RoBERTa)."""
    css_class = "model-icon-indobert" if model_name == "IndoBERT" else "model-icon-roberta"
    initial = "I" if model_name == "IndoBERT" else "R"
    return f'<div class="model-icon {css_class}">{initial}</div>'


def load_eval_metrics() -> dict:
    """
    Metrik evaluasi hasil training pada test set (lihat evaluation_result.png
    dari notebook training: learning curve, confusion matrix, ROC curve).
    """
    return {
        "IndoBERT": {
            "accuracy": 0.987,
            "precision": 0.987,
            "recall": 0.987,
            "f1": 0.987,
            "roc_auc": 1.000,
        },
        "RoBERTa": {
            "accuracy": 0.995,
            "precision": 0.995,
            "recall": 0.995,
            "f1": 0.995,
            "roc_auc": 0.999,
        },
    }