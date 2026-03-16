# KDS Platformu — ABI/IAD Entegrasyonu

## Dosya Yapısı

```
abi_analysis_test.py      ← Orijinal, hiç değiştirilmedi
kds_abi_engine.py         ← ABIAnalyzer'ı miras alan motor (yeni)
kds_python.py             ← KDS hasta motoru (önceki)
kds_streamlit_full.py     ← Tam platform Streamlit uygulaması (yeni)
requirements_full.txt     ← Tüm bağımlılıklar
```

## Kurulum

```bash
pip install -r requirements_full.txt
```

## Çalıştırma

```bash
streamlit run kds_streamlit_full.py
```

## Sayfa Haritası

| Sayfa | İçerik |
|-------|--------|
| 🏠 Genel Bakış | Hasta KDS özeti, VKİ + SCORE2 göstergesi, öneriler |
| ⚖️ VKİ Analizi | VKİ/BKO hesaplama, obezite risk faktörleri |
| 💗 Life's Essential 8 | AHA LE8 radar + puanlama |
| 🧬 Cinsiyet Riski | Erkek/Kadın KDS risk profilleri |
| 👥 Tüm Hastalar | Hasta tablosu + grafikler |
| 🔬 ABI/IAD Analizi | **Çalışma A** ve **B** bağımsız olarak analiz |
| 📊 ABI Karşılaştırma | İki çalışma yan yana + istatistik + raporlar |
| ➕ Yeni Hasta | Hasta kayıt formu |

## ABI/IAD Kullanım Akışı

1. **🔬 ABI/IAD Analizi** sayfasına gidin
2. **Çalışma A** sekmesinde:
   - İsim, N ve tohum girin (veya CSV yükleyin)
   - "Analizi Çalıştır" butonuna tıklayın
3. **Çalışma B** sekmesinde aynı adımları tekrarlayın
4. **📊 ABI Karşılaştırma** sayfasına geçin
   - Yan yana metrikler, overlay histogramlar, violin plotlar
   - Mann-Whitney U, Chi-kare, Cohen's d testleri
   - 3 bağımsız raporu indir (.md)

## Mimari

```
abi_analysis_test.ABIAnalyzer
        ↑ miras (değiştirilmedi)
kds_abi_engine.KDSABIEngine      ← Streamlit veri katmanı
        ↓ iki örnek
kds_abi_engine.ABIComparator     ← İstatistiksel karşılaştırma
        ↓
kds_streamlit_full.py            ← UI katmanı
```

## İndirilebilir Raporlar

Her çalışmadan ve karşılaştırmadan `.md` formatında rapor indirilir:
- Demografik özet tablosu
- ABI/IAD sınıflandırma dağılımları
- Normal aralık kapsamı (Genel + Cleveland Clinic)
- Cinsiyet, yaş grubu, boy regresyonu, VKİ/IAD korelasyonları
- İstatistiksel test sonuçları (karşılaştırma raporunda)

---
> ⚕️ Bilgilendirme amaçlıdır. Klinik kararlar için hekim değerlendirmesi gereklidir.
