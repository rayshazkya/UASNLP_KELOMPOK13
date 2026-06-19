import pandas as pd
import re

def strip_intro_cliches(text):
    """
    Memotong kalimat pengantar khas media sosial agar teks langsung 
    masuk ke inti klaim/narasi hoaksnya saja.
    """
    text = str(text)
    
    # 1. Hapus pola: "Beredar unggahan... beserta narasi/klaim:"
    text = re.sub(r'(?i)beredar\s+.*?(\bnarasi\b|\bklaim\b|\bteks\b)\s*[:\s]*', '', text)
    
    # 2. Hapus sisa-sisa tanda petik dekoratif di awal/akhir narasi yang tertinggal
    text = text.replace('“', '').replace('”', '').replace('"', '').replace('\'', '')
    
    # 3. Rapikan huruf kapital di awal kalimat jika ada yang rusak akibat pemotongan
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
        
    return text

def poles_dataset_akhir(file_input='dataset_bert_semantic_perfect.csv', file_output='dataset_indobert_final.csv'):
    print("⏳ Memulai proses penyelarasan akhir (Final Polish)...")
    
    try:
        df = pd.read_csv(file_input)
    except FileNotFoundError:
        print(f"❌ File {file_input} tidak ditemukan.")
        return

    total_awal = len(df)
    dataset_clean = []
    dropped_short = 0
    
    # Iterasi per 2 baris secara ketat untuk menjamin sinkronisasi pasangan
    for i in range(0, total_awal, 2):
        if i + 1 >= total_awal:
            break
            
        row_hoax = df.iloc[i]
        row_fact = df.iloc[i+1]
        
        # Validasi double-check pastikan urutannya benar (Ganjil=1, Genap=0)
        if row_hoax['label'] != 1 or row_fact['label'] != 0:
            # Jika urutan bergeser, cari baris berikutnya yang valid agar tidak merusak data
            continue
            
        # Bersihkan teks hoaks dari kalimat pengantar medsos
        hoax_text = strip_intro_cliches(row_hoax['text'])
        fact_text = str(row_fact['text']).strip()
        
        # FILTER KETAT: Buang jika ada teks yang terlalu pendek (di bawah 100 karakter)
        if len(hoax_text) < 100 or len(fact_text) < 100:
            dropped_short += 1
            continue
            
        # Simpan pasangan yang terbukti steril dan sinkron
        dataset_clean.append({'text': hoax_text, 'label': 1, 'source': row_hoax['source']})
        dataset_clean.append({'text': fact_text, 'label': 0, 'source': row_fact['source']})

    # Konversi ke DataFrame baru
    df_final = pd.DataFrame(dataset_clean)
    df_final.to_csv(file_output, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("🎯 PROSES FINAL POLISH SUKSES!")
    print("="*50)
    print(f"• Total Baris Input       : {total_awal} baris")
    print(f"• Dibuang (Teks < 100 char): {dropped_short * 2} baris")
    print(f"• Total Baris Siap Train  : {len(df_final)} baris")
    print("\n📊 Distribusi Kelas Akhir (Pasti Seimbang):")
    print(df_final['label'].value_counts())
    print(f"\n💾 File steril disimpan di: '{file_output}'")
    print("="*50)

if __name__ == "__main__":
    # Jalankan file ini menggunakan file hasil preprocessing kamu sebelumnya
    poles_dataset_akhir()