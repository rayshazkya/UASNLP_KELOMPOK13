"""
app.py
Dashboard demo HoaxBERT: deteksi hoaks berita politik berbahasa Indonesia
menggunakan IndoBERT dan RoBERTa (fine-tuned).

Jalankan dengan:
    streamlit run app.py
"""

import streamlit as st
from utils import predict, load_eval_metrics

st.set_page_config(
    page_title="HoaxBERT - Deteksi Hoaks Berita Politik",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state.history = []


# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("HoaxBERT")
    st.caption("Deteksi Hoaks Berita Politik Berbahasa Indonesia")

    mode = st.radio(
        "Pilih mode deteksi",
        ["IndoBERT", "RoBERTa", "Komparasi Keduanya"],
    )

    st.divider()
    st.markdown("**Kelompok 13**")
    st.caption("Shafa Disya Aulia, Raysha Tazkiya Rahim, Muadz Fauzi")
    st.caption("Pemrosesan Bahasa Alami - Informatika USK")


# ---------- HEADER ----------
st.title("Deteksi Hoaks Berita Politik")
st.write(
    "Masukkan judul atau isi berita di bawah ini, lalu sistem akan "
    "memprediksi apakah berita tersebut berpotensi hoaks atau tidak."
)

tab_deteksi, tab_performa = st.tabs(["Deteksi", "Performa Model"])


# ---------- TAB DETEKSI ----------
with tab_deteksi:
    text_input = st.text_area(
        "Teks berita",
        height=180,
        placeholder="Tempel judul atau isi berita di sini...",
    )

    detect_clicked = st.button("Deteksi", type="primary")

    if detect_clicked:
        if not text_input.strip():
            st.warning("Teks berita masih kosong, isi dulu sebelum deteksi.")
        else:
            if mode == "Komparasi Keduanya":
                col1, col2 = st.columns(2)
                result_indo = predict(text_input, "IndoBERT")
                result_roberta = predict(text_input, "RoBERTa")

                with col1:
                    st.subheader("IndoBERT")
                    label = result_indo["label"]
                    conf = result_indo["confidence"]
                    color = "red" if label == "hoaks" else "green"
                    st.markdown(f"Label: **:{color}[{label.upper()}]**")
                    st.progress(conf, text=f"Confidence: {conf:.1%}")

                with col2:
                    st.subheader("RoBERTa")
                    label = result_roberta["label"]
                    conf = result_roberta["confidence"]
                    color = "red" if label == "hoaks" else "green"
                    st.markdown(f"Label: **:{color}[{label.upper()}]**")
                    st.progress(conf, text=f"Confidence: {conf:.1%}")

                st.divider()
                if result_indo["label"] == result_roberta["label"]:
                    st.success(
                        f"Kedua model sepakat: berita ini diprediksi "
                        f"**{result_indo['label'].upper()}**."
                    )
                else:
                    st.error(
                        "Kedua model berbeda pendapat. "
                        f"IndoBERT: **{result_indo['label'].upper()}**, "
                        f"RoBERTa: **{result_roberta['label'].upper()}**."
                    )

                st.session_state.history.insert(
                    0,
                    {
                        "text": text_input[:80],
                        "indobert": result_indo["label"],
                        "roberta": result_roberta["label"],
                    },
                )

            else:
                result = predict(text_input, mode)
                label = result["label"]
                conf = result["confidence"]
                color = "red" if label == "hoaks" else "green"

                st.subheader("Hasil Prediksi")
                st.markdown(f"Label: **:{color}[{label.upper()}]**")
                st.progress(conf, text=f"Confidence: {conf:.1%}")

                st.session_state.history.insert(
                    0,
                    {
                        "text": text_input[:80],
                        mode.lower(): label,
                    },
                )

    if st.session_state.history:
        st.divider()
        st.subheader("Riwayat Prediksi (sesi ini)")
        for item in st.session_state.history[:5]:
            st.write(item)


# ---------- TAB PERFORMA MODEL ----------
with tab_performa:
    st.subheader("Perbandingan Performa Model")
    st.caption(
        "Angka di bawah ini diambil dari hasil evaluasi pada test set "
        "(lihat fungsi load_eval_metrics di utils.py untuk mengisi nilai asli)."
    )

    metrics = load_eval_metrics()
    col1, col2 = st.columns(2)

    for col, model_name in zip([col1, col2], metrics.keys()):
        with col:
            st.markdown(f"**{model_name}**")
            m = metrics[model_name]
            st.metric("Accuracy", f"{m['accuracy']:.2%}")
            st.metric("Precision", f"{m['precision']:.2%}")
            st.metric("Recall", f"{m['recall']:.2%}")
            st.metric("F1-score", f"{m['f1']:.2%}")
            st.metric("ROC-AUC", f"{m['roc_auc']:.2%}")

    st.divider()
    st.caption(
        "Tambahkan confusion matrix di sini dengan st.image('path/ke/gambar.png') "
        "kalau sudah disimpan dari notebook training."
    )
