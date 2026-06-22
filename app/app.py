"""
app.py
Dashboard demo HoaxBERT: deteksi hoaks berita politik berbahasa Indonesia
menggunakan IndoBERT dan RoBERTa (fine-tuned).

CATATAN PENTING soal styling:
Tag HTML pembungkus (mis. <div class="card">...</div>) HARUS dibangun sebagai
SATU string utuh lalu dirender lewat SATU pemanggilan st.markdown(). Kalau
tag pembuka dan penutup dipisah di dua st.markdown() berbeda dengan komponen
Streamlit asli (st.metric, st.text_area, dst) di antaranya, browser akan
menutup tag itu sendiri di setiap potongan render - hasilnya card kosong
melayang dan komponen di dalamnya jadi polos di luar card.

Untuk membungkus komponen Streamlit asli (text_area, button, columns),
pakai st.container(border=True) yang sudah didukung native oleh Streamlit -
jangan pakai trik div manual untuk kasus ini.

Jalankan dengan:
    streamlit run app.py
"""

import os

import streamlit as st
from utils import (
    predict,
    load_eval_metrics,
    render_result_card_html,
    render_metrics_card_html,
    hero_illustration_svg,
)

st.set_page_config(
    page_title="HoaxBERT - Deteksi Hoaks Berita Politik",
    layout="wide",
)

# ---------- LOAD CUSTOM CSS ----------
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("# HoaxBERT")
    st.caption("Deteksi Hoaks Berita Berbahasa Indonesia")

    st.markdown("&nbsp;")
    mode = st.radio(
        "MODE DETEKSI",
        ["IndoBERT", "RoBERTa", "Komparasi Keduanya"],
    )

    st.markdown("---")
    st.markdown("**Kelompok 13**")
    st.caption("Shafa Disya Aulia")
    st.caption("Raysha Tazkiya Rahim")
    st.caption("Muadz Fauzi")
    st.markdown("&nbsp;")
    st.caption("Pemrosesan Bahasa Alami — Informatika USK")


# ---------- HEADER ----------
st.markdown(
    f'<div class="hero">'
    f'<div class="hero-text">'
    f'<h1>Deteksi Hoaks Berita </h1>'
    f'<p>Masukkan judul atau isi berita di bawah ini, lalu sistem akan '
    f'memprediksi apakah berita tersebut berpotensi hoaks atau tidak.</p>'
    f'</div>'
    f'<div class="hero-illustration">{hero_illustration_svg()}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

tab_deteksi, tab_performa = st.tabs(["Deteksi", "Performa Model"])


# ---------- TAB DETEKSI ----------
with tab_deteksi:
    # Container bordered bawaan Streamlit -> aman membungkus text_area + button
    with st.container(border=True):
        st.markdown('<div class="card-header">Teks Berita</div>', unsafe_allow_html=True)
        text_input = st.text_area(
            "Teks berita",
            height=160,
            placeholder="Tempel judul atau isi berita di sini...",
            label_visibility="collapsed",
        )
        detect_clicked = st.button("Deteksi Sekarang", type="primary")

    if detect_clicked:
        if not text_input.strip():
            st.warning("Teks berita masih kosong, isi dulu sebelum deteksi.")
        else:
            if mode == "Komparasi Keduanya":
                result_indo = predict(text_input, "IndoBERT")
                result_roberta = predict(text_input, "RoBERTa")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        render_result_card_html("IndoBERT", result_indo),
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        render_result_card_html("RoBERTa", result_roberta),
                        unsafe_allow_html=True,
                    )

                if result_indo["label"] == result_roberta["label"]:
                    st.markdown(
                        f'<div class="verdict-agree">Kedua model sepakat: '
                        f'berita ini diprediksi {result_indo["label"].upper()}.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="verdict-disagree">Kedua model berbeda pendapat — '
                        f'IndoBERT: {result_indo["label"].upper()}, '
                        f'RoBERTa: {result_roberta["label"].upper()}.</div>',
                        unsafe_allow_html=True,
                    )

                st.session_state.history.insert(
                    0,
                    f'"{text_input[:70]}..." → IndoBERT: '
                    f'{result_indo["label"]}, RoBERTa: {result_roberta["label"]}',
                )

            else:
                result = predict(text_input, mode)
                st.markdown(
                    render_result_card_html(mode, result),
                    unsafe_allow_html=True,
                )

                st.session_state.history.insert(
                    0, f'"{text_input[:70]}..." → {mode}: {result["label"]}'
                )

    if st.session_state.history:
        st.markdown("#### Riwayat Prediksi (sesi ini)")
        for item in st.session_state.history[:5]:
            st.markdown(f'<div class="history-row">{item}</div>', unsafe_allow_html=True)


# ---------- TAB PERFORMA MODEL ----------
with tab_performa:
    st.markdown(
        "Angka di bawah ini diambil dari hasil evaluasi pada test set. "
        "Metrik yang ditampilkan: Accuracy, Precision, Recall, F1-score, dan ROC-AUC."
    )

    metrics = load_eval_metrics()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            render_metrics_card_html("IndoBERT", metrics["IndoBERT"]),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_metrics_card_html("RoBERTa", metrics["RoBERTa"]),
            unsafe_allow_html=True,
        )

    st.markdown("#### Visualisasi Hasil Evaluasi")
    st.markdown(
        "Learning curve, confusion matrix, dan ROC curve dari proses "
        "training & evaluasi kedua model pada test set."
    )

    image_path = "assets/evaluation_result.png"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning(
            f"Gambar belum ditemukan di `{image_path}`. "
            "Taruh file evaluation_result.png di folder app/assets/."
        )

    with st.container(border=True):
        st.markdown("**Cara membaca grafik ini:**")
        st.markdown(
            """
- **Learning Curve (kiri & tengah)** — menunjukkan Train Loss dan Val Loss
  menurun stabil sepanjang 5 epoch, sementara Val F1 terus meningkat hingga
  mendekati 1.0. Ini menandakan kedua model belajar dengan baik tanpa tanda
  overfitting yang signifikan (val loss tidak naik kembali).
- **Confusion Matrix (kiri bawah & tengah bawah)** — dari 556 data test,
  IndoBERT hanya salah memprediksi 7 data (2 false positive, 5 false negative),
  sedangkan RoBERTa hanya salah 3 data (1 false positive, 2 false negative).
- **Perbandingan Metrik (kanan atas)** — RoBERTa unggul tipis dari IndoBERT
  di Accuracy, Precision, Recall, dan F1-score (99.5% vs 98.7%), sementara
  IndoBERT punya ROC-AUC sedikit lebih tinggi (1.000 vs 0.999).
- **ROC Curve (kanan bawah)** — kurva kedua model menempel sangat dekat ke
  pojok kiri atas, menandakan kemampuan klasifikasi yang hampir sempurna
  dalam membedakan berita hoaks dan non-hoaks.
            """
        )
        st.caption(
            "Kesimpulan: kedua model sama-sama berperforma sangat tinggi pada "
            "dataset ini, dengan RoBERTa unggul tipis secara keseluruhan."
        )