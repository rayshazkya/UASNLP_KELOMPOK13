import cloudscraper  # Membantu bypass proteksi 502 / Cloudflare
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os  # Fitur proteksi untuk mengecek file lama

def scrape_turnbackhoax(start_page=101, end_page=1100):
    base_url = "https://turnbackhoax.id/articles?category=all&page="
    nama_file = 'dataset_hoax_rapi.csv'
    
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    
    print(f"🕵️ [TAHAP 1] Memulai scraping TurnBackHoax.id dari halaman {start_page} sampai {end_page}...")
    
    # Looping berjalan dari halaman start_page sampai end_page
    for page in range(start_page, end_page + 1):
        url = f"{base_url}{page}"
        print(f"\n📄 Membaca Halaman {page} -> {url}")
        
        scraped_data = [] # Reset penampung data setiap ganti halaman baru
        
        try:
            response = scraper.get(url, timeout=20)
            if response.status_code != 200:
                print(f"⚠️ Gagal memuat halaman {page} (Status: {response.status_code})")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Cari link artikel yang mengandung sub-folder '/articles/'
            article_links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if '/articles/' in href and href != 'https://turnbackhoax.id/articles/' and href != 'https://turnbackhoax.id/articles':
                    article_links.append(href)
            
            article_links = list(set(article_links))
            if not article_links:
                print(f"⚠️ Tidak menemukan link artikel di halaman {page}.")
                continue
                
            print(f"🔍 Menemukan {len(article_links)} link artikel unik di halaman {page}. Mulai ekstraksi...")
            
            for article_url in article_links:
                slug_nama = article_url.split('/')[-1]
                print(f"   🔗 Detail: {slug_nama[:40]}...")
                
                detail_data = scrape_detail_article(scraper, article_url)
                if detail_data:
                    scraped_data.append(detail_data)
                    
                time.sleep(2) # Jeda aman anti-banned per artikel
                
            # ================================================================
            # CHUNKING SYSTEM: Langsung amankan data halaman ini ke CSV
            # ================================================================
            if scraped_data:
                df_page = pd.DataFrame(scraped_data)
                
                # Filter Kriteria Opsi 2: Hanya ambil yang ada link berita resminya
                df_target_page = df_page[df_page['referred_news_urls'] != ""].copy()
                
                if not df_target_page.empty:
                    # mode='a' artinya data ditumpuk di baris paling bawah (Append)
                    # header=False agar judul kolom tidak ikut berulang di tengah file CSV
                    df_target_page.to_csv(nama_file, mode='a', index=False, sep=';', encoding='utf-8-sig', header=False)
                    print(f"💾 [PAGE {page}] BERHASIL AMANKAN {len(df_target_page)} ARTIKEL KE DALAM CSV!")
                else:
                    print(f"ℹ️ [PAGE {page}] Selesai dibaca, tapi tidak ada artikel yang lolos kriteria Opsi 2.")
                    
        except Exception as e:
            print(f"❌ Error Terjadi pada halaman {page}: {e}")
            print("Skrip akan otomatis mencoba melompat ke halaman berikutnya...")
            time.sleep(5)

def scrape_detail_article(scraper, url):
    try:
        res = scraper.get(url, timeout=15)
        if res.status_code != 200: return None
        
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 1. Ambil & Bersihkan Judul
        title_elem = soup.find('h1', class_='entry-title') or soup.find('h1')
        title = title_elem.text.strip() if title_elem else "No Title"
        title = re.sub(r'\s+', ' ', title).strip()
            
        # 2. Ambil & Bersihkan Konten (Narasi Hoaks)
        content_div = soup.find('div', class_='entry-content') or soup.find('div', class_='pf-content') or soup.find('article')
        if not content_div: return None
        
        # Regex untuk mengunci teks menjadi 1 baris lurus (menghapus enter, tab, spasi ganda)
        full_text = re.sub(r'\s+', ' ', content_div.text.strip()).strip()
        
        # 3. Ekstrak link eksternal portal berita resmi
        links = content_div.find_all('a', href=True)
        reference_links = []
        news_domains = ['kompas.com', 'detik.com', 'liputan6.com', 'tempo.co', 'merdeka.com', 'antaranews.com', 'republika.co.id', 'viva.co.id', 'kumparan.com', 'cnbcindonesia.com']
        
        for link in links:
            href = link['href']
            if any(domain in href.lower() for domain in news_domains):
                reference_links.append(href)
                
        reference_links = list(set(reference_links))
        
        # Gabungkan list link menjadi string biasa dipisahkan koma agar rapi di CSV
        reference_links_clean = ", ".join(reference_links) if reference_links else ""
        
        return {
            'tbh_title': title,
            'tbh_url': url,
            'hoax_raw_text': full_text,
            'referred_news_urls': reference_links_clean
        }
        
    except Exception as e:
        return None

if __name__ == "__main__":
    nama_file = 'dataset_hoax_rapi.csv'
    
    # 🛡️ LOGIKA PROTEKSI OTOMATIS:
    # Mengecek apakah file hasil halaman 1-100 sudah ada di folder
    if not os.path.exists(nama_file):
        print("🧹 File CSV belum ada. Menyiapkan file baru dengan header kolom...")
        df_headers = pd.DataFrame(columns=['tbh_title', 'tbh_url', 'hoax_raw_text', 'referred_news_urls'])
        df_headers.to_csv(nama_file, index=False, sep=';', encoding='utf-8-sig')
    else:
        print(f"📦 File '{nama_file}' ditemukan!")
        print("Sistem akan langsung MENYAMBUNGKAN data baru tepat di bawah baris halaman 1-100.")
    
    # JALANKAN REVISI LANJUTAN: MULAI DARI HALAMAN 628 SAMPAI 1700
    scrape_turnbackhoax(start_page=628, end_page=1700)
    
    print(f"\n🎉 SELESAI TOTAL! Semua data dari halaman 628-1700 yang lolos kriteria berhasil ditumpuk rapi di '{nama_file}'")