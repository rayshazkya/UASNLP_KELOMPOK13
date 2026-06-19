import pandas as pd
import re

def clean_and_format_hoax(text):
    """
    Membedah teks hoaks: memotong penjelasan, mengambil judul klaim & narasi,
    serta merombak total gaya bahasanya agar menyerupai portal berita formal.
    """
    text = str(text)
    
    # 1. AMPUTASI: Buang bagian penjelasan ke bawah agar tidak terjadi kebocoran fakta
    if "Penjelasan" in text:
        text = text.split("Penjelasan")[0].strip()
        
    # 2. METADATA STRIPPING: Pisahkan Judul Klaim dengan Deskripsi Narasi
    # Pola TurnBackHoax: [SALAH] Judul Teks <Kategori> <Tanggal> Mafindo Narasi <Isi>
    match = re.search(r'^\[.*?\]\s*(.*?)\s+\b\w+\s+\d{2}/\d{2}/\d{4}\s+Mafindo\s+Narasi\s+(.*)$', text)
    
    if match:
        title = match.group(1).strip()
        body = match.group(2).strip()
        combined_text = f"{title}. {body}"
    else:
        # Fallback jika pola reguler sedikit bergeser
        combined_text = re.sub(r'^\[.*?\]\s*', '', text)
        combined_text = re.sub(r'\s+Mafindo\s+Narasi\s+', '. ', combined_text)

    # 3. NORMALISASI GAYA: Buang noise platform & standarisasi teks
    combined_text = re.sub(r'\[arsip\]', '', combined_text, flags=re.IGNORECASE)
    combined_text = re.sub(r'Hingga.*?(tanda suka|komentar|dibagikan).*$', '', combined_text) # Buang statistik medsos di akhir narasi
    
    # CRUSHER CAPS LOCK: Ubah kata-kata yang menggunakan HURUF KAPITAL BERUNTUN menjadi huruf kecil
    combined_text = re.sub(r'\b[A-Z]{4,}\b', lambda m: m.group(0).lower(), combined_text)
    
    # Rapikan spasi ganda
    combined_text = re.sub(r'\s+', ' ', combined_text).strip()
    return combined_text

def match_fact_length(fact_text, target_length):
    """
    Memotong teks berita fakta pada batas kalimat terdekat 
    agar panjang karakternya sebanding dengan teks hoaks pasangannya.
    """
    fact_text = str(fact_text).strip()
    if len(fact_text) <= target_length:
        return fact_text
        
    # Pecah teks fakta berdasarkan tanda baca akhir kalimat
    sentences = re.split(r'(?<=[.!?])\s+', fact_text)
    current_text = ""
    
    for sentence in sentences:
        # Beri toleransi kelebihan panjang sedikit (+150 karakter) agar kalimat tidak gantung
        if len(current_text) + len(sentence) > target_length + 150:
            break
        current_text += (sentence + " ")
        
    # Jika pemecahan kalimat gagal, lakukan hard-truncate aman
    if not current_text.strip():
        return fact_text[:target_length].strip()
        
    return current_text.strip()

def eksekusi_filtering():
    file_input = 'dataset_bert_mixed_perfect.csv'
    file_output = 'dataset_bert_semantic_perfect.csv'
    
    print("📦 Membaca dataset hasil scraping...")
    df = pd.read_csv(file_input)
    
    dataset_final = []
    total_rows = len(df)
    drop_short_fact = 0
    
    print(f"⚙️ Memproses {total_rows} baris dengan metode Penyandingan Gaya Semantik...")
    
    # Looping melompati data per 2 baris (Pasangan Hoaks-Fakta)
    for i in range(0, total_rows, 2):
        if i + 1 >= total_rows:
            break
            
        row_hoax = df.iloc[i]
        row_fakta = df.iloc[i+1]
        
        # 1. Bersihkan dan samakan gaya penulisan teks Hoaks
        hoax_clean = clean_and_format_hoax(row_hoax['text'])
        fakta_raw = str(row_fakta['text']).strip()
        
        # 2. Validasi kelayakan teks fakta (buang yang gagal scrape / terlalu pendek)
        if len(fakta_raw) < 400:
            drop_short_fact += 1
            continue
            
        # 3. SETARAKAN PANJANG: Potong teks fakta mengikuti panjang hoaks yang sudah bersih
        fakta_clean = match_fact_length(fakta_raw, len(hoax_clean))
        
        # 4. Masukkan kembali sebagai pasangan yang seimbang (1:1)
        dataset_final.append({'text': hoax_clean, 'label': 1, 'source': 'TurnBackHoax_Cleaned'})
        dataset_final.append({'text': fakta_clean, 'label': 0, 'source': row_fakta['source']})

    # Simpan ke file baru
    if dataset_final:
        df_final = pd.DataFrame(dataset_final)
        df_final.to_csv(file_output, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print("🎉 PREPROCESSING SELESAI DAN SUKSES TOTAL!")
        print("="*50)
        print(f"• Dataset Awal  : {total_rows} baris")
        print(f"• Dibuang (Fakta pendek): {drop_short_fact * 2} baris")
        print(f"• Dataset Baru  : {len(df_final)} baris (Seimbang 50:50)")
        print(f"• File Output   : '{file_output}'")
        print("-" * 50)
        print("💡 Karakteristik Sekarang: Kedua kelas kini sama-sama berwujud")
        print("   narasi berita formal, panjang teks per pasangan sangat identik,")
        print("   dan bebas dari kata bocoran [SALAH]/Mafindo.")
        print("==================================================")
    else:
        print("❌ Gagal memproses data. Periksa kembali struktur file inputmu.")

if __name__ == "__main__":
    eksekusi_filtering()