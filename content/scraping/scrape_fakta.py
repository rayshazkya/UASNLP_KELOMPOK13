import pandas as pd
import cloudscraper
from newspaper import Article  # Library "Cheat" untuk ekstrak berita otomatis
import time
import re
import os

def scrape_fakta_jalur_cheat(file_csv_hoax):
    print("🔄 [TAHAP 2 - SAFE CHEAT MODE] Membaca file hasil Tahap 1...")
    
    # 1. Validasi Input
    try:
        df = pd.read_csv(file_csv_hoax, sep=';')
    except FileNotFoundError:
        print(f"❌ File '{file_csv_hoax}' tidak ditemukan. Jalankan Tahap 1 terlebih dahulu!")
        return
    
    # Pecah string link yang dipisahkan koma menjadi list asli Python, lalu di-explode ke bawah
    df['referred_news_urls'] = df['referred_news_urls'].fillna('').apply(lambda x: [url.strip() for url in x.split(',') if url.strip()])
    df_exploded = df.explode('referred_news_urls').dropna(subset=['referred_news_urls'])
    df_exploded = df_exploded[df_exploded['referred_news_urls'] != '']
    
    if df_exploded.empty:
        print("❌ Tidak ada link berita yang bisa diproses di dalam CSV.")
        return

    # 2. Inisialisasi File Output & Sistem Checkpoint
    nama_file_final = 'dataset_bert_mixed_perfect.csv'
    log_checkpoint = 'processed_fakta_urls.txt'
    
    # Jika file output belum ada, buat file kosong beserta headernya terlebih dahulu
    if not os.path.exists(nama_file_final):
        df_headers = pd.DataFrame(columns=['text', 'label', 'source'])
        df_headers.to_csv(nama_file_final, index=False, encoding='utf-8-sig')
        print(f"🧹 Membuat file output baru: '{nama_file_final}'")
    
    # Baca riwayat URL yang sudah sukses di-scrape sebelumnya (biar bisa resume otomatis)
    urls_terproses = set()
    if os.path.exists(log_checkpoint):
        with open(log_checkpoint, 'r') as f:
            urls_terproses = set(line.strip() for line in f if line.strip())
        print(f"📦 Menemukan riwayat: {len(urls_terproses)} URL sudah pernah sukses diproses sebelumnya.")

    # 3. Setup Scraper Pembobol Proteksi
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    
    print(f"🚀 Total target: {len(df_exploded)} link fakta. Mulai mengekstrak secara dicicil...")
    sukses_count = 0
    
    # 4. Looping Ekstraksi Data
    for index, row in df_exploded.iterrows():
        url_berita = row['referred_news_urls']
        
        # JALUR AMAN: Lewati jika URL ini sudah pernah sukses diproses pada sesi sebelumnya
        if url_berita in urls_terproses:
            continue
            
        print(f"🔗 Memproses otomatis: {url_berita}")
        
        try:
            # Ambil HTML mentah menggunakan cloudscraper (aman dari blokir)
            res = scraper.get(url_berita, timeout=15)
            if res.status_code != 200:
                print(f"   ⚠️ Gagal akses portal (Status: {res.status_code})")
                continue
                
            # JALUR CHEAT: Inisialisasi Newspaper3k dengan bahasa Indonesia
            artikel_otomatis = Article(url_berita, language='id')
            artikel_otomatis.set_html(res.text)
            artikel_otomatis.parse()
            
            teks_berita = artikel_otomatis.text
            
            # Pembersihan ekstra: Ubah enter (\n) dan spasi ganda menjadi satu baris lurus
            teks_berita = re.sub(r'\s+', ' ', teks_berita).strip()
            
            # Validasi apakah teks berita berhasil diambil dan memiliki panjang yang cukup
            if teks_berita and len(teks_berita) > 100:
                # Siapkan sepasang data (Hoaks & Fakta)
                data_pasangan = [
                    {'text': row['hoax_raw_text'], 'label': 1, 'source': 'TurnBackHoax'},
                    {'text': teks_berita, 'label': 0, 'source': url_berita.split('/')[2]}
                ]
                
                # LANGSUNG AMANKAN KE CSV (Append Mode 'a')
                df_chunk = pd.DataFrame(data_pasangan)
                df_chunk.to_csv(nama_file_final, mode='a', index=False, encoding='utf-8-sig', header=False)
                
                # Catat URL ini ke file txt checkpoint agar tidak di-scrape ulang jika sistem restart
                with open(log_checkpoint, 'a') as f:
                    f.write(url_berita + '\n')
                
                urls_terproses.add(url_berita)
                sukses_count += 1
                print("   ✅ SUKSES! Konten berhasil diekstrak dan dicicil ke CSV.")
            else:
                print("   ⚠️ Konten gagal diekstrak secara otomatis atau terlalu pendek.")
                
            # Jeda etis 2 detik
            time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Error saat memproses link ini: {e}")
            time.sleep(3) # Beri jeda nafas sedikit jika terjadi error network
            
    print("\n====================================================")
    print(f"🎉 PROSES SELESAI!")
    print(f"Berhasil menambahkan {sukses_count * 2} baris data baru pada sesi ini.")
    print(f"Seluruh dataset kumulatif aman di: '{nama_file_final}'")
    print("====================================================")

if __name__ == "__main__":
    # Pastikan file 'dataset_hoax_rapi.csv' hasil dari file Tahap 1 sudah berada di folder yang sama
    scrape_fakta_jalur_cheat('dataset_hoax_rapi.csv')