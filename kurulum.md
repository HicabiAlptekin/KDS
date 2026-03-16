# KDS Platformu — Kurulum Kılavuzu

## SADECE BU 2 DOSYA GEREKLİ

```
kds_platform.py          ← Ana uygulama (tek dosya, self-contained)
abi_analysis_test.py     ← Opsiyonel: CLI ile ABI analizi
```

`kds_platform.py` tek başına çalışır — başka Python dosyasına ihtiyaç duymaz.

---

## KURULUM

### Adım 1 — Python paketlerini kur
```bash
pip install streamlit plotly pandas numpy scipy scikit-learn seaborn matplotlib tabulate
```

### Adım 2 — Uygulamayı çalıştır
```bash
streamlit run kds_platform.py
```

Tarayıcınızda otomatik açılır: http://localhost:8501

---

## SORUN GİDERME

### "ModuleNotFoundError: 'kds_python'" hatası
→ Eski dosyayı (`kds_streamlit_full.py`) silip sadece `kds_platform.py` kullanın.

### "ModuleNotFoundError: 'streamlit'" hatası
→ `pip install streamlit` komutunu çalıştırın.

### Python 3.11 uyumluluğu
→ Tam uyumlu. Python 3.9+ gerektirir.
