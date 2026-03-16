#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KDS + ABI/IAD Platformu v3 — Tek Dosya, Self-Contained
=======================================================
YENİ ÖZELLİKLER (v3):
  - ABI/IAD testi hasta kayıtlarından otomatik veri alır (Çalışma B kalktı)
  - Hasta başına ABI/IAD hesaplama + KDS karşılaştırması
  - İki test aynı yönü gösteriyorsa: ŞİDDETLİ TAVSİYELER
  - Farklı yön gösteriyorsa: TEĞİT + GEREKÇELİ TAKİP KARARLARI
  - Hasta takip dosyası (geçmiş test kayıtları, oturum başına JSON-benzeri)
  - Sağlık eğrisi: zaman içinde metrik değişimi
  - Neyin iyiye / kötüye gittiği raporu

Kurulum:
    pip install streamlit plotly pandas numpy scipy scikit-learn tabulate

Çalıştır:
    streamlit run kds_platform_v3.py
"""

import warnings, math, json, copy
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from datetime import datetime, date
from scipy import stats

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — VERİ MODELLERİ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Hasta:
    ad: str
    soyad: str
    yas: int
    cinsiyet: str
    boy_cm: float
    kilo_kg: float
    bel_cm: float
    kalca_cm: float
    sistolik_kb: int
    diyastolik_kb: int
    total_kolesterol: float
    hdl: float
    ldl: float
    aclik_kan_sekeri: float
    hba1c: float
    trigliserid: float
    sigara: str
    aktivite_dk_hafta: int
    uyku_saat: float
    diyabet: str
    # ABI/IAD ölçüm alanları (hasta kaydında saklanır)
    sbp_sag_kol: float   = 125.0
    sbp_sol_kol: float   = 122.0
    sbp_sag_ayak: float  = 130.0
    sbp_sol_ayak: float  = 128.0
    # Kadına özel
    menopoz: Optional[str] = None
    pcos: bool = False
    gebelik_komplikasyon: Optional[str] = None
    hrt: bool = False
    tarih: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    hasta_id: str = field(default_factory=lambda: f"H{datetime.now().strftime('%Y%m%d%H%M%S')}")


@dataclass
class TestKaydi:
    """Tek bir test oturumunun anlık görüntüsü — sağlık eğrisi için."""
    hasta_id: str
    tarih: str
    # KDS metrikleri
    kds_le8_toplam: int
    kds_score2: float
    kds_risk_sinifi: str
    kds_ckm_evresi: int
    kds_le8_detay: dict
    # ABI/IAD metrikleri
    abi_sag: float
    abi_sol: float
    iad_mmhg: float
    abi_sinifi: str
    iad_sinifi: str
    # Klinik değerler (anlık)
    kilo_kg: float
    vki: float
    sistolik_kb: int
    total_kolesterol: float
    hba1c: float
    # Özel notlar
    notlar: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — KDS HESAPLAMA MOTORLERİ
# ══════════════════════════════════════════════════════════════════════════════

class VKIHesaplayici:
    @staticmethod
    def hesapla(boy_cm, kilo_kg):
        m = boy_cm / 100
        return round(kilo_kg / m**2, 1)

    @staticmethod
    def siniflandir(v):
        if v < 18.5: return "Zayıf"
        if v < 25:   return "Normal"
        if v < 30:   return "Fazla Kilolu"
        if v < 35:   return "Obez I"
        if v < 40:   return "Obez II"
        return "Obez III"

    @staticmethod
    def bko(bel, kalca): return round(bel / kalca, 2)

    @staticmethod
    def bko_risk(bko, cins): return bko > (0.90 if cins == "E" else 0.85)

    @staticmethod
    def bel_riski(bel, cins):
        if cins == "E":
            if bel < 94:   return "Normal"
            if bel < 102:  return "Artmış Risk"
            return "Yüksek Risk"
        if bel < 80:   return "Normal"
        if bel < 88:   return "Artmış Risk"
        return "Yüksek Risk"

    @staticmethod
    def ideal_kilo(boy_cm, cins):
        m = boy_cm / 100
        return round(18.5*m*m, 1), round(24.9*m*m, 1)

    @staticmethod
    def le8_skoru(v):
        if v < 18.5: return 40
        if v < 23:   return 100
        if v < 25:   return 90
        if v < 27:   return 75
        if v < 30:   return 60
        if v < 35:   return 35
        if v < 40:   return 20
        return 5


class LE8Hesaplayici:
    @staticmethod
    def aktivite_skoru(dk):
        if dk >= 150: return 100
        if dk >= 100: return 80
        if dk >= 60:  return 60
        if dk >= 30:  return 40
        if dk >= 10:  return 20
        return 0

    @staticmethod
    def nikotin_skoru(s):
        return 100 if s == "hayir" else 70 if s == "birakti" else 0

    @staticmethod
    def uyku_skoru(s):
        if 7 <= s <= 9:                  return 100
        if 6 <= s < 7 or 9 < s <= 10:   return 70
        if 5 <= s < 6 or 10 < s <= 11:  return 40
        return 10

    @staticmethod
    def kb_skoru(sbp, dbp):
        if sbp < 120 and dbp < 80: return 100
        if sbp < 130: return 80
        if sbp < 140: return 60
        if sbp < 160: return 40
        return 10

    @staticmethod
    def lipid_skoru(tc, hdl):
        r = tc / hdl if hdl > 0 else 10
        if r < 3.5:  return 100
        if r < 4.5:  return 80
        if r < 5.5:  return 60
        if r < 6.5:  return 40
        return 20

    @staticmethod
    def glisemi_skoru(glu, hba1c, diab):
        if diab in ("tip1","tip2"):    return 30
        if glu < 100 and hba1c < 5.7: return 100
        if glu < 110 or hba1c < 6.0:  return 75
        if glu < 126 or hba1c < 6.5:  return 50
        return 25

    @classmethod
    def hesapla(cls, h: Hasta) -> dict:
        vki = VKIHesaplayici.hesapla(h.boy_cm, h.kilo_kg)
        d = {
            "Diyet":                 65,
            "Fiziksel Aktivite":     cls.aktivite_skoru(h.aktivite_dk_hafta),
            "Nikotin":               cls.nikotin_skoru(h.sigara),
            "Uyku":                  cls.uyku_skoru(h.uyku_saat),
            "Vücut Ağırlığı (VKİ)": VKIHesaplayici.le8_skoru(vki),
            "Lipidler":              cls.lipid_skoru(h.total_kolesterol, h.hdl),
            "Glisemi":               cls.glisemi_skoru(h.aclik_kan_sekeri, h.hba1c, h.diyabet),
            "Kan Basıncı":           cls.kb_skoru(h.sistolik_kb, h.diyastolik_kb),
        }
        return {"detay": d, "toplam": round(sum(d.values())/8)}


class SCORE2Hesaplayici:
    @staticmethod
    def hesapla(h: Hasta) -> float:
        vki  = VKIHesaplayici.hesapla(h.boy_cm, h.kilo_kg)
        risk = 3.0
        if h.yas > 70:    risk += 8
        elif h.yas > 65:  risk += 6
        elif h.yas > 60:  risk += 4
        elif h.yas > 55:  risk += 2
        if h.cinsiyet == "E": risk += 2
        if h.sistolik_kb >= 180:    risk += 6
        elif h.sistolik_kb >= 160:  risk += 4
        elif h.sistolik_kb >= 140:  risk += 3
        elif h.sistolik_kb >= 130:  risk += 1
        if h.total_kolesterol >= 310:   risk += 4
        elif h.total_kolesterol >= 270: risk += 3
        elif h.total_kolesterol >= 230: risk += 2
        elif h.total_kolesterol >= 200: risk += 1
        if h.sigara == "evet":    risk += 5
        elif h.sigara == "birakti": risk += 1
        if h.diyabet in ("tip1","tip2"): risk += 4
        elif h.diyabet == "prediyabet":  risk += 1
        if vki >= 35:    risk += 4
        elif vki >= 30:  risk += 3
        elif vki >= 27:  risk += 1
        if h.cinsiyet == "K" and h.menopoz == "post": risk += 2
        if h.cinsiyet == "K" and h.pcos:              risk += 1
        return round(min(risk, 45), 1)

    @staticmethod
    def siniflandir(p):
        if p < 5:    return "Düşük"
        if p < 10:   return "Orta"
        if p < 20:   return "Yüksek"
        return "Çok Yüksek"


class CKMDegerlendirici:
    @staticmethod
    def evre(h: Hasta, vki: float) -> int:
        s = 0
        if vki >= 30:    s += 2
        elif vki >= 27:  s += 1
        if h.diyabet in ("tip1","tip2"): s += 2
        elif h.diyabet == "prediyabet":  s += 1
        if h.sistolik_kb >= 140: s += 1
        if h.ldl >= 160:         s += 1
        if s == 0: return 0
        if s <= 2: return 1
        if s <= 4: return 2
        return 3


class GLP1Degerlendirici:
    @staticmethod
    def endike_mi(vki, h: Hasta):
        if vki < 27:
            return False, "VKİ eşiğin altında (< 27). Yaşam tarzı değişiklikleri önceliklidir."
        rf = []
        if h.sistolik_kb >= 130:      rf.append("hipertansiyon")
        if h.diyabet != "yok":         rf.append("diyabet/pre-diyabet")
        if h.total_kolesterol >= 200:  rf.append("dislipidemi")
        if h.sigara == "evet":         rf.append("sigara")
        if rf:
            return True, f"GLP-1RA değerlendirilebilir. Risk faktörleri: {', '.join(rf)}. Doktor onayı zorunludur."
        return False, "VKİ eşiğin üzerinde ancak ek KVH risk faktörü saptanmadı."


def tam_degerlendirme(h: Hasta) -> dict:
    vki  = VKIHesaplayici.hesapla(h.boy_cm, h.kilo_kg)
    bko  = VKIHesaplayici.bko(h.bel_cm, h.kalca_cm)
    le8  = LE8Hesaplayici.hesapla(h)
    s2   = SCORE2Hesaplayici.hesapla(h)
    ckm  = CKMDegerlendirici.evre(h, vki)
    ge, gm = GLP1Degerlendirici.endike_mi(vki, h)
    return {
        "hasta": h, "vki": {
            "deger": vki, "sinif": VKIHesaplayici.siniflandir(vki),
            "bko": bko, "bko_risk": VKIHesaplayici.bko_risk(bko, h.cinsiyet),
            "bel_riski": VKIHesaplayici.bel_riski(h.bel_cm, h.cinsiyet),
            "ideal_min": VKIHesaplayici.ideal_kilo(h.boy_cm, h.cinsiyet)[0],
            "ideal_max": VKIHesaplayici.ideal_kilo(h.boy_cm, h.cinsiyet)[1],
        },
        "le8": le8, "score2": {"yuzde": s2, "sinif": SCORE2Hesaplayici.siniflandir(s2)},
        "ckm_evresi": ckm, "glp1": {"endike": ge, "mesaj": gm},
    }


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — ABI/IAD MOTORu (HASTA BAZLI)
# ══════════════════════════════════════════════════════════════════════════════

class HastaABIHesaplayici:
    """Hasta kayıtlarındaki tansiyon verilerinden ABI/IAD hesaplar."""

    @staticmethod
    def hesapla(h: Hasta) -> dict:
        abi_sag = round(h.sbp_sag_ayak / h.sbp_sag_kol, 3) if h.sbp_sag_kol > 0 else float("nan")
        abi_sol = round(h.sbp_sol_ayak / h.sbp_sol_kol, 3) if h.sbp_sol_kol > 0 else float("nan")
        iad     = round(abs(h.sbp_sag_kol - h.sbp_sol_kol), 1)

        def cls_abi(x):
            if math.isnan(x): return "Bilinmiyor"
            if x < 0.9:       return "PAH Riski"
            if x > 1.4:       return "Kalsifikasyon Riski"
            if x < 1.0:       return "Sınırda Düşük"
            return "Normal"

        def cls_iad(i):
            if i > 20: return "Ölçüm Hatası Olasılığı"
            if i >= 10: return "Uyarı (≥10 mmHg)"
            return "Normal"

        return {
            "abi_sag":      abi_sag,
            "abi_sol":      abi_sol,
            "iad":          iad,
            "abi_sinif":    cls_abi(abi_sag),
            "iad_sinif":    cls_iad(iad),
            "sbp_sag_kol":  h.sbp_sag_kol,
            "sbp_sol_kol":  h.sbp_sol_kol,
            "sbp_sag_ayak": h.sbp_sag_ayak,
            "sbp_sol_ayak": h.sbp_sol_ayak,
        }

    @staticmethod
    def kds_risk_seviyesi(abi_sinif: str) -> str:
        """ABI sınıfını KDS risk seviyesine çevirir."""
        if abi_sinif == "PAH Riski":              return "Yüksek"
        if abi_sinif == "Kalsifikasyon Riski":    return "Yüksek"
        if abi_sinif == "Sınırda Düşük":          return "Orta"
        return "Düşük"


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — ÇAPRAZ KARŞILAŞTIRMA & KARAR MOTORU
# ══════════════════════════════════════════════════════════════════════════════

class CaprazKarsilastirmaMotoru:
    """KDS ve ABI/IAD sonuçlarını karşılaştırır, karar üretir."""

    # Risk seviyesi → sayısal (yüksek = kötü)
    _RISK_SEVIYE = {"Düşük": 0, "Orta": 1, "Yüksek": 2, "Çok Yüksek": 3}

    @classmethod
    def karsilastir(cls, kds_sonuc: dict, abi_sonuc: dict) -> dict:
        kds_risk   = kds_sonuc["score2"]["sinif"]
        abi_risk   = HastaABIHesaplayici.kds_risk_seviyesi(abi_sonuc["abi_sinif"])
        kds_s      = cls._RISK_SEVIYE.get(kds_risk, 0)
        abi_s      = cls._RISK_SEVIYE.get(abi_risk, 0)

        ayni_yon   = (kds_s >= 1 and abi_s >= 1) or (kds_s == 0 and abi_s == 0)
        fark       = abs(kds_s - abi_s)
        dominant   = "KDS" if kds_s >= abi_s else "ABI"

        # Genel risk seviyesi
        combined   = max(kds_s, abi_s)
        if kds_s >= 1 and abi_s >= 1:
            combined = kds_s + abi_s  # her ikisi de yüksek → toplayarak ağırlıkla
        combined_sinif = ["Düşük","Orta","Yüksek","Çok Yüksek"][min(combined, 3)]

        tavsiyeler = cls._tavsiye_uret(kds_sonuc, abi_sonuc, kds_risk, abi_risk,
                                       ayni_yon, kds_s, abi_s)
        return {
            "kds_risk":       kds_risk,
            "abi_risk":       abi_risk,
            "kds_seviye":     kds_s,
            "abi_seviye":     abi_s,
            "ayni_yon":       ayni_yon,
            "fark":           fark,
            "dominant":       dominant,
            "combined_sinif": combined_sinif,
            "tavsiyeler":     tavsiyeler,
        }

    @classmethod
    def _tavsiye_uret(cls, kds, abi_res, kds_risk, abi_risk, ayni_yon, ks, as_) -> list:
        h          = kds["hasta"]
        vki        = kds["vki"]["deger"]
        le8        = kds["le8"]["toplam"]
        s2         = kds["score2"]["yuzde"]
        abi_val    = abi_res["abi_sag"]
        iad_val    = abi_res["iad"]
        tavsiyeler = []

        if ayni_yon and ks >= 1 and as_ >= 1:
            # ─── İKİ TEST DE YÜKSEK RİSK GÖSTERİYOR → ŞİDDETLİ TAVSİYELER ────
            tavsiyeler.append({
                "tip":    "ŞİDDETLİ",
                "baslik": "ACİL KARDİYOLOJİ KONSÜLTASYONU",
                "mesaj":  (
                    f"KDS (SCORE2 %{s2}, {kds_risk}) ve ABI ({abi_val:.3f}, {abi_res['abi_sinif']}) testleri "
                    f"aynı yönde yüksek risk işaret etmektedir. Bu uyum, gerçek periferik arter hastalığı veya "
                    f"sistemik ateroskleroz riskini güçlü biçimde doğrulamaktadır. "
                    f"7 gün içinde kardiyoloji değerlendirmesi zorunludur."
                ),
                "eylem":  "🏥 KARDİYOLOJİ SEVK — BUGÜN"
            })
            if vki >= 30:
                tavsiyeler.append({
                    "tip":    "ŞİDDETLİ",
                    "baslik": "OBEZİTE + DAMAR HASTALĞI KOMBİNASYONU",
                    "mesaj":  (
                        f"VKİ {vki} ({kds['vki']['sinif']}) ile düşük ABI, aterosklerotik yük artışının "
                        f"belirgin göstergesidir. SELECT RCT verilerine göre semaglutide gibi GLP-1RA ajanlar "
                        f"bu profilde ASCVD riskini %20 azaltabilir. Endokrinoloji ve kardiyoloji ortak değerlendirmesi."
                    ),
                    "eylem": "💊 GLP-1RA DEĞERLENDİRMESİ — ACELE"
                })
            if h.sistolik_kb >= 140:
                tavsiyeler.append({
                    "tip":    "ŞİDDETLİ",
                    "baslik": "HİPERTANSİYON + PAH YÜKSEK RİSK",
                    "mesaj":  (
                        f"Sistolik KB {h.sistolik_kb} mmHg ile birlikte düşük ABI ({abi_val:.3f}), "
                        f"periferik arterlerde ciddi ateroskleroz yükü anlamına gelir. "
                        f"Antihipertansif tedavi acilen optimize edilmeli; ACE inhibitörü veya ARB "
                        f"periferik dolaşımı koruyucu tercih edilmelidir."
                    ),
                    "eylem": "💉 ANTİHİPERTANSİF OPTİMİZASYON — HAFTAYA"
                })
            if h.sigara == "evet":
                tavsiyeler.append({
                    "tip":    "ŞİDDETLİ",
                    "baslik": "SİGARA + PAH — DAMAR YIKIMI DEVAM EDİYOR",
                    "mesaj":  (
                        "Sigara içimi, periferik arter hastalığının en güçlü modifiye edilebilir risk "
                        "faktörüdür. Aktif sigara + düşük ABI kombinasyonu amputasyon riskini 10 kat artırır. "
                        "Vareniklin destekli bırakma programı derhal başlatılmalıdır."
                    ),
                    "eylem": "🚭 SİGARA BIRAKMA — BUGÜN BAŞLA"
                })
            if iad_val >= 10:
                tavsiyeler.append({
                    "tip":    "ŞİDDETLİ",
                    "baslik": "İAD ≥10 mmHg + YÜK RİSK — SUBKLAVYEN STEN ARAŞTIR",
                    "mesaj":  (
                        f"Kollar arası {iad_val:.1f} mmHg fark, mevcut yüksek kardiyovasküler risk "
                        f"bağlamında subklavyen stenoz veya aterosklerotik obstrüksiyonu akla getirir. "
                        f"Doppler USG ile subklavyen ve karotid değerlendirme önerilir."
                    ),
                    "eylem": "🔬 DOPPLER USG — 2 HAFTA"
                })
            if le8 < 50:
                tavsiyeler.append({
                    "tip":    "ŞİDDETLİ",
                    "baslik": "LE8 SKORU KRİTİK DÜŞÜK + DAMARSAL HASAR",
                    "mesaj":  (
                        f"LE8 skoru {le8}/100 ile tüm temel sağlık faktörleri yetersiz düzeyde. "
                        f"Damarsal hasar bu tabloda geri döndürülemez hasara ilerleme riski taşır. "
                        f"Multidisipliner sağlık planı (kardiyoloji, diyetisyen, fizyoterapist) oluşturulmalı."
                    ),
                    "eylem": "📋 MULTİDİSİPLİNER PLAN — 1 AY"
                })

        elif ayni_yon and ks == 0 and as_ == 0:
            # ─── İKİ TEST DE DÜŞÜK RİSK → OLUMLU ONAY ──────────────────────
            tavsiyeler.append({
                "tip":    "OLUMLU",
                "baslik": "İKİ TEST UYUMU: DÜŞÜK KARDİYOVASKÜLER RİSK",
                "mesaj":  (
                    f"KDS (SCORE2 %{s2}) ve ABI ({abi_val:.3f}) testleri birlikte düşük risk "
                    f"doğrulamaktadır. Mevcut sağlıklı yaşam tarzının sürdürülmesi önerilir. "
                    f"Yıllık rutin tarama yeterlidir."
                ),
                "eylem": "✅ YILLIK TARAMA — RUTIN"
            })

        else:
            # ─── TESTLER FARKLI YÖN GÖSTERİYOR → TEĞİT TAKİP KARARLARI ─────
            tavsiyeler.append({
                "tip":    "TEĞİT",
                "baslik": "TEST UYUMSUZLUĞU — DİKKATLİ YORUMLAMA GEREKİYOR",
                "mesaj":  (
                    f"KDS SCORE2 %{s2} ({kds_risk}) ve ABI {abi_val:.3f} ({abi_res['abi_sinif']}) "
                    f"farklı risk yönleri göstermektedir. Bu durum, sistemik risk ile lokal vasküler "
                    f"durum arasındaki olası ayrışmayı yansıtabilir. Tek başına karar vermek yanıltıcı olabilir."
                ),
                "eylem": "⚠️ DİKKATLİ DEĞERLENDİRME"
            })

            if ks > as_:  # KDS yüksek, ABI normal
                tavsiyeler.append({
                    "tip":    "TEĞİT",
                    "baslik": "KDS RİSK YÜKSEK / ABI NORMAL — SİSTEMİK RİSK ÖN PLANDA",
                    "mesaj":  (
                        f"SCORE2 %{s2} yüksek kardiyometabolik risk gösterirken ABI {abi_val:.3f} "
                        f"periferik dolaşımın henüz korunduğuna işaret ediyor. Bu pencere, "
                        f"müdahale için değerli bir erken dönemdir. ABI kötüleşmeden önce "
                        f"kardiyometabolik risk faktörleri agresif yönetilmelidir."
                    ),
                    "eylem": "🎯 ERKEİN MÜDAHALE PENCERESİ — 3 AY"
                })
                if h.diyabet in ("tip2","prediyabet"):
                    tavsiyeler.append({
                        "tip":    "TEĞİT",
                        "baslik": "DİYABET + YÜKSEK KDS — SESSIZ PAH TARAMA",
                        "mesaj":  (
                            "Diyabetik hastalarda periferik nöropati ABI'yi yanlış yüksek gösterebilir "
                            "(medial kalsifikasyon). Diyabetik hastalarda ABI >1.3 de patolojik kabul edilir. "
                            "Ayak bileği-kol basınç indeksini toe-brachial index (TBI) ile doğrulayın."
                        ),
                        "eylem": "🦶 TBI ÖLÇÜMÜ — 1 AY"
                    })
                if h.total_kolesterol >= 230:
                    tavsiyeler.append({
                        "tip":    "TEĞİT",
                        "baslik": "YÜKSEK KOLESTEROl — PLAK YÜKÜ İZLEMİ",
                        "mesaj":  (
                            f"Total kolesterol {h.total_kolesterol} mg/dL. Yüksek lipit yükü subklinik "
                            f"ateroskleroz başlatabilir; ABI henüz etkilenmemiş olsa da karotid intima-media "
                            f"kalınlığı (CIMT) ölçümü ve yıllık ABI takibi önerilir."
                        ),
                        "eylem": "🩺 CIMT + ABI YILLIK TAKİP"
                    })

            else:  # ABI yüksek risk, KDS düşük/orta
                tavsiyeler.append({
                    "tip":    "TEĞİT",
                    "baslik": "ABI ANORMALİ / KDS DÜŞÜK — LOKALİZE VASKÜLER SORUN",
                    "mesaj":  (
                        f"ABI {abi_val:.3f} ({abi_res['abi_sinif']}) periferik vasküler patoloji "
                        f"işaret ederken sistemik KDS riski görece düşük. Bu tablo; izole periferik "
                        f"arter hastalığı, bacak yaralanma öyküsü veya teknik ölçüm hatası olabilir. "
                        f"İkinci ölçüm farklı koşulda ve vasküler cerrahi konsültasyonu ile doğrulanmalı."
                    ),
                    "eylem": "🔬 VASKÜLER CERRAHİ KONSÜLTASYON — 1 AY"
                })
                if iad_val >= 10:
                    tavsiyeler.append({
                        "tip":    "TEĞİT",
                        "baslik": "İAD ≥10 mmHg — TEK TARAFLI OBSTRÜKSYON OLASILIğI",
                        "mesaj":  (
                            f"IAD {iad_val:.1f} mmHg asimetrik bulgusuna karşın sistemik risk düşük. "
                            f"Subklavyen veya aksiller arterde tek taraflı stenoz veya tromboz düşünülmeli. "
                            f"Renkli Doppler USG ile bilateral kol değerlendirmesi."
                        ),
                        "eylem": "🩺 DOPPLER USG — 2 HAFTA"
                    })

        return tavsiyeler


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — SAĞLIK EĞRİSİ & TAKİP SİSTEMİ
# ══════════════════════════════════════════════════════════════════════════════

class SaglikEgrisiMotoru:
    """Hasta geçmişinden sağlık eğrisi üretir."""

    @staticmethod
    def kayit_olustur(h: Hasta, kds_sonuc: dict, abi_sonuc: dict, not_: str = "") -> TestKaydi:
        return TestKaydi(
            hasta_id      = h.hasta_id,
            tarih         = datetime.now().strftime("%Y-%m-%d %H:%M"),
            kds_le8_toplam= kds_sonuc["le8"]["toplam"],
            kds_score2    = kds_sonuc["score2"]["yuzde"],
            kds_risk_sinifi= kds_sonuc["score2"]["sinif"],
            kds_ckm_evresi= kds_sonuc["ckm_evresi"],
            kds_le8_detay = kds_sonuc["le8"]["detay"],
            abi_sag       = abi_sonuc["abi_sag"],
            abi_sol       = abi_sonuc["abi_sol"],
            iad_mmhg      = abi_sonuc["iad"],
            abi_sinifi    = abi_sonuc["abi_sinif"],
            iad_sinifi    = abi_sonuc["iad_sinif"],
            kilo_kg       = h.kilo_kg,
            vki           = VKIHesaplayici.hesapla(h.boy_cm, h.kilo_kg),
            sistolik_kb   = h.sistolik_kb,
            total_kolesterol= h.total_kolesterol,
            hba1c         = h.hba1c,
            notlar        = not_,
        )

    @staticmethod
    def egri_dataframe(gecmis: List[TestKaydi]) -> pd.DataFrame:
        if not gecmis:
            return pd.DataFrame()
        rows = []
        for k in gecmis:
            rows.append({
                "Tarih":        k.tarih,
                "LE8":          k.kds_le8_toplam,
                "SCORE2%":      k.kds_score2,
                "Risk":         k.kds_risk_sinifi,
                "CKM":          k.kds_ckm_evresi,
                "ABI Sağ":      k.abi_sag,
                "IAD (mmHg)":   k.iad_mmhg,
                "ABI Sınıfı":   k.abi_sinifi,
                "Kilo (kg)":    k.kilo_kg,
                "VKİ":          k.vki,
                "Sistolik KB":  k.sistolik_kb,
                "Kolesterol":   k.total_kolesterol,
                "HbA1c":        k.hba1c,
                "Notlar":       k.notlar,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def degisim_raporu(gecmis: List[TestKaydi]) -> dict:
        """İlk ve son kayıt arasındaki değişimi hesaplar."""
        if len(gecmis) < 2:
            return {}
        ilk, son = gecmis[0], gecmis[-1]
        metrikler = {
            "LE8 Skoru":     (ilk.kds_le8_toplam, son.kds_le8_toplam, "yüksek_iyi"),
            "SCORE2 Riski":  (ilk.kds_score2,      son.kds_score2,     "dusuk_iyi"),
            "ABI (Sağ)":     (ilk.abi_sag,          son.abi_sag,         "1_e_yakin_iyi"),
            "IAD (mmHg)":    (ilk.iad_mmhg,         son.iad_mmhg,        "dusuk_iyi"),
            "Kilo (kg)":     (ilk.kilo_kg,           son.kilo_kg,         "dusuk_iyi"),
            "VKİ":           (ilk.vki,               son.vki,             "dusuk_iyi"),
            "Sistolik KB":   (ilk.sistolik_kb,       son.sistolik_kb,     "dusuk_iyi"),
            "Kolesterol":    (ilk.total_kolesterol,  son.total_kolesterol,"dusuk_iyi"),
            "HbA1c":         (ilk.hba1c,             son.hba1c,           "dusuk_iyi"),
            "CKM Evresi":    (ilk.kds_ckm_evresi,    son.kds_ckm_evresi,  "dusuk_iyi"),
        }
        sonuc = {"iyiye": [], "kotye": [], "stabil": []}
        for ad, (bslk, son_, yon) in metrikler.items():
            delta = son_ - bslk
            if abs(delta) < 0.01:
                sonuc["stabil"].append({"metrik": ad, "baslangic": bslk, "son": son_, "delta": 0})
                continue
            if yon == "yüksek_iyi":
                hedef = "iyiye" if delta > 0 else "kotye"
            elif yon == "dusuk_iyi":
                hedef = "iyiye" if delta < 0 else "kotye"
            else:  # 1_e_yakin_iyi (ABI için 1.0-1.3 ideal)
                hedef = "iyiye" if abs(son_ - 1.1) < abs(bslk - 1.1) else "kotye"
            sonuc[hedef].append({"metrik": ad, "baslangic": bslk, "son": son_,
                                  "delta": round(delta, 3)})
        sonuc["toplam_kayit"] = len(gecmis)
        sonuc["ilk_tarih"]    = ilk.tarih
        sonuc["son_tarih"]    = son.tarih
        return sonuc


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — DEMO VERİLER
# ══════════════════════════════════════════════════════════════════════════════

def _demo_hastalar():
    return [
        Hasta("Ahmet","Kaya",   58,"E",175,94, 102,110,145,92, 210,38,138,108,6.8,195,"evet",  60, 6.5,"prediyabet",
              sbp_sag_kol=148,sbp_sol_kol=144,sbp_sag_ayak=122,sbp_sol_ayak=118,
              hasta_id="H001"),
        Hasta("Fatma","Yıldız", 65,"K",162,78,  91,105,155,95, 225,42,148,105,6.2,210,"hayir", 45, 7.0,"tip2",
              sbp_sag_kol=158,sbp_sol_kol=155,sbp_sag_ayak=135,sbp_sol_ayak=130,
              menopoz="post",hasta_id="H002"),
        Hasta("Ayşe", "Demir",  45,"K",165,63,  76,100,118,76, 185,58,112, 88,5.4,120,"hayir",180, 7.5,"yok",
              sbp_sag_kol=120,sbp_sol_kol=118,sbp_sag_ayak=125,sbp_sol_ayak=122,
              menopoz="pre",hasta_id="H003"),
        Hasta("Mehmet","Arslan",72,"E",170,98, 108,112,158,98, 240,35,162,115,7.2,240,"evet",  30, 5.5,"tip2",
              sbp_sag_kol=162,sbp_sol_kol=155,sbp_sag_ayak=128,sbp_sol_ayak=124,
              hasta_id="H004"),
        Hasta("Ali",  "Çelik",  55,"E",178,86,  97,106,138,88, 210,42,135, 95,5.7,165,"hayir", 90, 7.0,"prediyabet",
              sbp_sag_kol=140,sbp_sol_kol=137,sbp_sag_ayak=138,sbp_sol_ayak=135,
              hasta_id="H005"),
    ]

def _demo_gecmis(hasta_id: str) -> List[TestKaydi]:
    """Belirli bir hasta için örnek geçmiş kayıtlar üretir."""
    import random
    random.seed(int(hasta_id.replace("H","")) * 7)
    rng = np.random.default_rng(int(hasta_id.replace("H","")) * 7)
    kayitlar = []
    for i in range(4):
        ay_once = 4 - i
        ilerleme = i * 0.08
        kayit = TestKaydi(
            hasta_id       = hasta_id,
            tarih          = f"2025-{(3 + (4-ay_once)):02d}-15 10:00",
            kds_le8_toplam = int(48 + i*5 + rng.integers(-3,4)),
            kds_score2     = round(max(5, 18 - i*2.5 + rng.normal(0,1)), 1),
            kds_risk_sinifi= ["Yüksek","Yüksek","Orta","Orta"][i],
            kds_ckm_evresi = max(0, 2 - i//2),
            kds_le8_detay  = {
                "Diyet":65,"Fiziksel Aktivite":35+i*8,"Nikotin":20,
                "Uyku":60+i*5,"Vücut Ağırlığı (VKİ)":35+i*6,
                "Lipidler":55+i*4,"Glisemi":45+i*5,"Kan Basıncı":40+i*6
            },
            abi_sag        = round(0.72 + i*0.04 + rng.normal(0,0.01), 3),
            abi_sol        = round(0.70 + i*0.04 + rng.normal(0,0.01), 3),
            iad_mmhg       = round(max(3, 12 - i*2 + rng.normal(0,1)), 1),
            abi_sinifi     = ["PAH Riski","PAH Riski","Sınırda Düşük","Sınırda Düşük"][i],
            iad_sinifi     = ["Uyarı (≥10 mmHg)","Uyarı (≥10 mmHg)","Uyarı (≥10 mmHg)","Normal"][i],
            kilo_kg        = round(95 - i*1.5, 1),
            vki            = round(VKIHesaplayici.hesapla(175, 95 - i*1.5), 1),
            sistolik_kb    = int(152 - i*4),
            total_kolesterol= round(215 - i*5, 1),
            hba1c          = round(6.9 - i*0.2, 1),
            notlar         = ["İlk değerlendirme","Diyet programı başlandı",
                              "İlaç dozaj artırımı","Kontrol muayenesi"][i],
        )
        kayitlar.append(kayit)
    return kayitlar


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7 — STREAMLIT SAYFA AYARLARI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="KDS Platformu v3", page_icon="❤️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main .block-container{padding-top:1.5rem}
h1{font-size:1.5rem!important;font-weight:600!important}
h2{font-size:1.15rem!important;font-weight:500!important}
h3{font-size:.95rem!important;font-weight:500!important}
.kds-metric{background:#f8f9fa;border-radius:10px;padding:1rem 1.2rem;margin-bottom:.5rem;border:1px solid #e9ecef}
.kds-metric .label{font-size:.72rem;color:#6c757d;margin-bottom:3px}
.kds-metric .value{font-size:1.5rem;font-weight:600}
.kds-metric .sub{font-size:.7rem;color:#6c757d;margin-top:2px}
/* Tavsiye kartları */
.tv-siddetli{border-left:5px solid #dc2626;background:#fef2f2;border-radius:8px;padding:.9rem 1.1rem;margin-bottom:.7rem}
.tv-tegit    {border-left:5px solid #d97706;background:#fffbeb;border-radius:8px;padding:.9rem 1.1rem;margin-bottom:.7rem}
.tv-olumlu  {border-left:5px solid #16a34a;background:#f0fdf4;border-radius:8px;padding:.9rem 1.1rem;margin-bottom:.7rem}
.tv-baslik  {font-weight:600;font-size:.9rem;margin-bottom:.3rem}
.tv-mesaj   {font-size:.83rem;line-height:1.6;color:#374151}
.tv-eylem   {font-size:.78rem;font-weight:600;margin-top:.4rem;color:#1e3a5f}
/* iyiye/kötüye renkleri */
.iyi-badge  {background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:99px;font-size:.72rem;font-weight:500}
.kotu-badge {background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:99px;font-size:.72rem;font-weight:500}
.stab-badge {background:#f3f4f6;color:#374151;padding:2px 8px;border-radius:99px;font-size:.72rem;font-weight:500}
/* Hasta listesi hover */
.hasta-row{padding:.5rem .3rem;border-bottom:1px solid #f0f0f0;cursor:pointer}
.hasta-row:hover{background:#f9fafb;border-radius:6px}
/* Hasta başlık bandı */
.hasta-baslik{background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);
  border-radius:12px;padding:.9rem 1.3rem;margin-bottom:1rem;color:#fff}
.hasta-baslik .isim{font-size:1.25rem;font-weight:700;letter-spacing:.01em}
.hasta-baslik .bilgi{font-size:.78rem;opacity:.85;margin-top:3px}
.hasta-baslik .test-bilgi{font-size:.75rem;opacity:.75;margin-top:5px;
  background:rgba(255,255,255,.12);border-radius:6px;padding:4px 8px;display:inline-block}
/* Test tarihi satırları */
.test-satir{display:flex;align-items:center;gap:8px;padding:6px 0;
  border-bottom:1px solid var(--color-border-tertiary,#eee)}
.test-satir:last-child{border-bottom:none}
.test-no{background:#3b82f6;color:#fff;border-radius:99px;
  font-size:.68rem;font-weight:600;padding:2px 8px;min-width:28px;text-align:center}
.test-tarih{font-size:.82rem;font-weight:500;flex:1}
.test-gun{font-size:.75rem;color:#6b7280}
.test-gecen{background:#f0fdf4;color:#065f46;border-radius:99px;
  font-size:.7rem;padding:2px 8px;font-weight:500}
.test-gecen-uyari{background:#fef3c7;color:#92400e;border-radius:99px;
  font-size:.7rem;padding:2px 8px;font-weight:500}
/* Terim sözlüğü */
.sozluk-baslik{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;
  color:#9ca3af;margin:1.5rem 0 .6rem;font-weight:600}
.sozluk-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px;margin-bottom:1rem}
.sozluk-kart{background:#f8f9fa;border:1px solid #e9ecef;border-radius:8px;
  padding:.65rem .85rem}
.sozluk-terim{font-size:.82rem;font-weight:600;color:#1e3a5f;margin-bottom:3px}
.sozluk-aciklama{font-size:.76rem;color:#4b5563;line-height:1.55}
.sozluk-deger{font-size:.72rem;color:#6b7280;margin-top:3px;
  background:#e9ecef;border-radius:4px;padding:1px 6px;display:inline-block}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR (SESSION STATE'ten ÖNCE — tüm sayfalar kullanır)
# ══════════════════════════════════════════════════════════════════════════════

# ── Tarih yardımcıları ────────────────────────────────────────────────────────

def _tarih_parse(tarih_str: str):
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(tarih_str.strip(), fmt)
        except ValueError:
            continue
    return None

def _tarih_goster(tarih_str: str) -> str:
    dt = _tarih_parse(tarih_str)
    return dt.strftime("%d %b %Y") if dt else tarih_str[:10]

def _gun_farki(tarih_str: str) -> int:
    dt = _tarih_parse(tarih_str)
    if dt is None: return -1
    return (datetime.now() - dt).days

def _gun_farki_str(tarih_str: str) -> str:
    g = _gun_farki(tarih_str)
    if g < 0:   return "tarih bilinmiyor"
    if g == 0:  return "bugün"
    if g == 1:  return "dün"
    if g < 7:   return f"{g} gün önce"
    if g < 30:  return f"{g//7} hafta önce"
    if g < 365: return f"{g//30} ay önce"
    return f"{g//365} yıl önce"

def _iki_tarih_arasi_gun(t1: str, t2: str) -> int:
    d1, d2 = _tarih_parse(t1), _tarih_parse(t2)
    if d1 and d2: return abs((d2 - d1).days)
    return -1

def test_tarihleri_html(gecmis_listesi: list) -> str:
    if not gecmis_listesi:
        return '<p style="font-size:.82rem;color:#9ca3af">Henüz kayıt yok.</p>'
    rows = ""
    for i, k in enumerate(gecmis_listesi):
        tarih_str = _tarih_goster(k.tarih)
        gun_str   = _gun_farki_str(k.tarih)
        if i > 0:
            ara = _iki_tarih_arasi_gun(gecmis_listesi[i-1].tarih, k.tarih)
            ara_html = f'<span class="test-gecen">{ara} gün sonra</span>' if ara >= 0 else ""
        else:
            ara_html = '<span class="test-gecen-uyari">İlk kayıt</span>'
        not_k = f' — <em style="color:#9ca3af;font-size:.72rem">{k.notlar}</em>' if k.notlar else ""
        rows += (f'<div class="test-satir">'
                 f'<span class="test-no">#{i+1}</span>'
                 f'<span class="test-tarih">{tarih_str}{not_k}</span>'
                 f'<span class="test-gun">{gun_str}</span>'
                 f'{ara_html}</div>')
    return f'<div style="margin:.5rem 0">{rows}</div>'

def hasta_baslik_html(h, gecmis_listesi: list) -> str:
    cins_tr = "Erkek" if h.cinsiyet == "E" else "Kadın"
    if gecmis_listesi:
        son_k = gecmis_listesi[-1]
        test_bilgi = (f"📋 {len(gecmis_listesi)} kayıt  ·  "
                      f"Son test: {_tarih_goster(son_k.tarih)}  ·  {_gun_farki_str(son_k.tarih)}")
    else:
        test_bilgi = "📋 Henüz test kaydı yok"
    return (f'<div class="hasta-baslik">'
            f'<div class="isim">👤 {h.ad} {h.soyad}</div>'
            f'<div class="bilgi">{h.yas} yaş · {cins_tr} · {h.hasta_id} · Kayıt: {h.tarih}</div>'
            f'<div class="test-bilgi">{test_bilgi}</div>'
            f'</div>')

# ── Terim sözlüğü ────────────────────────────────────────────────────────────

TERIMLER: dict = {
    "VKİ (Vücut Kitle İndeksi)": {
        "aciklama": "Boyunuza göre kilonuzun uygun olup olmadığını gösteren basit bir ölçüt. Kilo (kg) ÷ Boy² (m²) formülü ile hesaplanır.",
        "deger": "Normal: 18.5–24.9 · Fazla kilolu: 25–29.9 · Obez: ≥30"
    },
    "LE8 (Life's Essential 8)": {
        "aciklama": "Amerikan Kalp Derneği (AHA) tarafından geliştirilen 8 temel yaşam alışkanlığını puanlayan kalp sağlığı skoru. Diyet, egzersiz, uyku, sigara, kilo, kolesterol, kan şekeri ve kan basıncını içerir. Ne kadar yüksekse o kadar iyi.",
        "deger": "İdeal: 80-100 · Orta: 50-79 · Düşük: 0-49"
    },
    "SCORE2": {
        "aciklama": "Avrupa'da geliştirilmiş, yaş, cinsiyet, tansiyon, kolesterol ve sigara bilgilerini kullanarak önümüzdeki 10 yılda kalp krizi veya inme geçirme olasılığını yüzde olarak tahmin eden risk hesabı.",
        "deger": "Düşük: <%5 · Orta: %5–10 · Yüksek: %10–20 · Çok yüksek: >%20"
    },
    "ABI (Ayak Bileği-Kol İndeksi)": {
        "aciklama": "Bacaklardaki kan akışının ne kadar sağlıklı olduğunu gösteren ölçüm. Ayak bileği tansiyonunu kol tansiyonuna bölerek elde edilir. Düşük çıkması, bacak damarlarında yağ birikmesi nedeniyle tıkanıklık oluştuğuna işaret eder.",
        "deger": "Normal: 1.0–1.3 · Sınırda: 0.9–1.0 · PAH Riski: <0.9 · Kalsifikasyon: >1.4"
    },
    "IAD (Kollar Arası Sistolik Fark)": {
        "aciklama": "İki kolda ölçülen büyük tansiyon değerleri arasındaki farktır. Bu farkın büyük olması, kollardan birine giden damar yolunda daralma ya da tıkanıklık olabileceğini düşündürür.",
        "deger": "Normal: <10 mmHg · Uyarı: 10–20 mmHg · Olası hata/patoloji: >20 mmHg"
    },
    "PAH (Periferik Arter Hastalığı)": {
        "aciklama": "Bacak ya da kol damarlarında yağ-kolesterol birikmesi (ateroskleroz) nedeniyle kan akışının azalmasıdır. Yürürken bacakta ağrı, yara iyileşmemesi, soğukluk PAH belirtisi olabilir.",
        "deger": "ABI <0.9 ise PAH riski yüksektir; <0.7 şiddetli hastalığa işaret eder"
    },
    "Lipidler (Kan Yağları)": {
        "aciklama": "Kandaki kolesterol ve trigliserid gibi yağ maddelerinin genel adıdır. LDL 'kötü kolesterol' olarak bilinir — damar duvarında birikerek tıkanıklığa yol açar. HDL ise 'iyi kolesterol' olarak damar duvarını temizler.",
        "deger": "LDL <100 mg/dL ideal · HDL >40 mg/dL (erkek), >50 mg/dL (kadın) · Trigliserid <150"
    },
    "Glisemi (Kan Şekeri)": {
        "aciklama": "Kandaki şeker (glikoz) miktarıdır. Açlık kan şekerinin yüksek olması veya son 3 ayın ortalamasını gösteren HbA1c'nin yüksek çıkması diyabet ya da prediyabet işaretidir. Kontrolsüz kan şekeri damarları ve sinirleri zamanla tahrip eder.",
        "deger": "Açlık: <100 mg/dL normal · 100–125 prediyabet · ≥126 diyabet"
    },
    "Sistolik KB (Kan Basıncı)": {
        "aciklama": "Kalp her attığında damarlara uygulanan basıncın en yüksek değeridir — halk arasında 'büyük tansiyon' denir. Sürekli yüksek olması damarları zorlar, kalp krizini ve inmeyi tetikleyebilir.",
        "deger": "İdeal: <120 mmHg · Normal: 120–129 · Yüksek tansiyon: ≥130 mmHg"
    },
    "CKM Sendromu": {
        "aciklama": "Kalp-damar (Kardiyovasküler), Böbrek ve Metabolizma hastalıklarının birbiriyle bağlantılı olarak birlikte ilerlediği durumu tanımlar. Obezite → diyabet → böbrek hasarı → kalp hastalığı zinciri tipik örüntüdür.",
        "deger": "Evre 0: Risk yok · Evre 1: Risk faktörü var · Evre 2: Hasar başlamış · Evre 3: İleri hasar"
    },
    "GLP-1RA (Semaglutide vb.)": {
        "aciklama": "Obezite ve diyabet tedavisinde kullanılan, kilo vermeye ve kan şekerini düzenlemeye yardımcı modern ilaç grubudur. SELECT araştırması (2023), bu ilaçların aynı zamanda kalp-damar olaylarını da azalttığını göstermiştir.",
        "deger": "Endikasyon: VKİ ≥27 + en az 1 KVH risk faktörü — doktor kararı şart"
    },
    "Bel/Kalça Oranı (BKO)": {
        "aciklama": "Bel çevresinin kalça çevresine bölünmesiyle bulunur. Karın bölgesindeki iç organ çevresindeki (visseral) yağlanmayı gösterir; bu yağlanma türü kalp hastalığı, diyabet ve hipertansiyonla güçlü bağlantılıdır.",
        "deger": "Erkek risk eşiği: >0.90 · Kadın risk eşiği: >0.85"
    },
    "Ateroskleroz": {
        "aciklama": "Damar duvarlarına kolesterol ve yağın birikmesi, ardından zamanla sertleşip damarın daralması sürecidir. Kalp krizi ve inmenin temel nedenidir. Genellikle belirtisiz ilerler; ABI, tahlil ve görüntülemelerle erken tespit edilebilir.",
        "deger": "Risk faktörleri: sigara, hipertansiyon, yüksek kolesterol, diyabet, obezite"
    },
    "HbA1c": {
        "aciklama": "Son 2–3 ayın ortalama kan şekerini gösteren kan testidir. Günlük açlık ölçümüne kıyasla çok daha güvenilir bir diyabet takip göstergesidir çünkü uzun dönem kontrolü yansıtır.",
        "deger": "<%5.7 normal · %5.7–6.4 prediyabet · ≥%6.5 diyabet tanısı"
    },
    "Menopoz Sonrası KDS Riski": {
        "aciklama": "Menopoz ile birlikte östrojen hormonu azalır. Östrojen damarları koruyan, kolesterolü düzenleyen bir hormondur; bu dönemden sonra kadınlarda kalp-damar hastalığı riski erkeklere yakın düzeye hızla yükselir.",
        "deger": "Menopoz sonrası KDS riski %40 oranında artış gösterebilir"
    },
}

SAYFA_TERİMLERİ = {
    "🏠 Genel Bakış":           ["VKİ (Vücut Kitle İndeksi)","LE8 (Life's Essential 8)","SCORE2","ABI (Ayak Bileği-Kol İndeksi)","CKM Sendromu"],
    "🔬 ABI/IAD Analizi":       ["ABI (Ayak Bileği-Kol İndeksi)","IAD (Kollar Arası Sistolik Fark)","PAH (Periferik Arter Hastalığı)","Ateroskleroz","Sistolik KB (Kan Basıncı)"],
    "⚖️ Çapraz Karşılaştırma": ["SCORE2","ABI (Ayak Bileği-Kol İndeksi)","IAD (Kollar Arası Sistolik Fark)","GLP-1RA (Semaglutide vb.)","CKM Sendromu"],
    "📈 Sağlık Eğrisi & Takip": ["LE8 (Life's Essential 8)","SCORE2","ABI (Ayak Bileği-Kol İndeksi)","Sistolik KB (Kan Basıncı)","HbA1c"],
    "💗 Life's Essential 8":    ["LE8 (Life's Essential 8)","Lipidler (Kan Yağları)","Glisemi (Kan Şekeri)","Sistolik KB (Kan Basıncı)","VKİ (Vücut Kitle İndeksi)"],
    "🧬 Cinsiyet Riski":        ["Menopoz Sonrası KDS Riski","Ateroskleroz","Bel/Kalça Oranı (BKO)","GLP-1RA (Semaglutide vb.)","PAH (Periferik Arter Hastalığı)"],
    "👥 Tüm Hastalar":          ["VKİ (Vücut Kitle İndeksi)","SCORE2","ABI (Ayak Bileği-Kol İndeksi)","LE8 (Life's Essential 8)","Ateroskleroz"],
    "➕ Yeni Hasta":            ["VKİ (Vücut Kitle İndeksi)","ABI (Ayak Bileği-Kol İndeksi)","IAD (Kollar Arası Sistolik Fark)","Glisemi (Kan Şekeri)","Sistolik KB (Kan Basıncı)"],
}

def sozluk_goster(sayfa_adi: str):
    terimler = SAYFA_TERİMLERİ.get(sayfa_adi, list(TERIMLER.keys())[:5])
    st.markdown('<div class="sozluk-baslik">📖 Bu Sayfada Geçen Terimlerin Açıklaması</div>',
                unsafe_allow_html=True)
    kartlar = ""
    for t in terimler:
        if t in TERIMLER:
            bilgi = TERIMLER[t]
            deger_html = (f'<div class="sozluk-deger">{bilgi["deger"]}</div>'
                          if bilgi.get("deger") else "")
            kartlar += (f'<div class="sozluk-kart">'
                        f'<div class="sozluk-terim">{t}</div>'
                        f'<div class="sozluk-aciklama">{bilgi["aciklama"]}</div>'
                        f'{deger_html}</div>')
    st.markdown(f'<div class="sozluk-grid">{kartlar}</div>', unsafe_allow_html=True)
    with st.expander("📚 Tüm Terimlerin Tam Sözlüğü"):
        for t, bilgi in TERIMLER.items():
            st.markdown(f"**{t}**")
            st.caption(bilgi["aciklama"])
            if bilgi.get("deger"):
                st.caption(f"*Referans değerler:* {bilgi['deger']}")
            st.divider()

# ── Grafik yardımcıları ───────────────────────────────────────────────────────

def vki_rengi(v):
    if v < 18.5: return "#3b82f6"
    if v < 25:   return "#22c55e"
    if v < 30:   return "#f59e0b"
    if v < 35:   return "#ef4444"
    return "#991b1b"

def risk_rengi(s):
    return {"Düşük":"#16a34a","Orta":"#d97706","Yüksek":"#dc2626","Çok Yüksek":"#7f1d1d"}.get(s,"#6b7280")

def le8_rengi(s):
    return "#22c55e" if s>=70 else "#f59e0b" if s>=50 else "#ef4444"

def abi_rengi(v):
    if math.isnan(v): return "#9ca3af"
    if v < 0.9:  return "#dc2626"
    if v < 1.0:  return "#f59e0b"
    if v <= 1.3: return "#16a34a"
    return "#dc2626"

def oncelik_cls(o):
    return {"Kritik":"rec-kritik","Yüksek":"rec-yuksek","Orta":"rec-orta","Bilgi":"rec-bilgi"}.get(o,"rec-bilgi")

def tavsiye_cls(tip):
    return {"ŞİDDETLİ":"tv-siddetli","TEĞİT":"tv-tegit","OLUMLU":"tv-olumlu"}.get(tip,"tv-tegit")

def tavsiye_html(tv):
    cls  = tavsiye_cls(tv["tip"])
    ikon = {"ŞİDDETLİ":"🚨","TEĞİT":"⚠️","OLUMLU":"✅"}.get(tv["tip"],"ℹ️")
    return (f'<div class="{cls}">'
            f'<div class="tv-baslik">{ikon} {tv["baslik"]}</div>'
            f'<div class="tv-mesaj">{tv["mesaj"]}</div>'
            f'<div class="tv-eylem">{tv["eylem"]}</div>'
            f'</div>')

def score2_gauge(pct):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={"suffix":"%","font":{"size":20}},
        gauge={"axis":{"range":[0,40]},
               "bar":{"color":"#ef4444" if pct>=10 else "#f59e0b" if pct>=5 else "#22c55e"},
               "steps":[{"range":[0,5],"color":"#d1fae5"},{"range":[5,10],"color":"#fef3c7"},
                        {"range":[10,20],"color":"#fee2e2"},{"range":[20,40],"color":"#fecaca"}]},
        title={"text":"SCORE2","font":{"size":12}}))
    fig.update_layout(height=190,margin=dict(t=30,b=5,l=15,r=15),paper_bgcolor="rgba(0,0,0,0)")
    return fig

def abi_gauge(val, label="ABI"):
    if math.isnan(val): val = 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={"font":{"size":20},"valueformat":".3f"},
        gauge={"axis":{"range":[0,2]},"bar":{"color":abi_rengi(val),"thickness":.3},
               "steps":[{"range":[0,0.9],"color":"#fee2e2"},
                        {"range":[0.9,1.0],"color":"#fef3c7"},
                        {"range":[1.0,1.3],"color":"#d1fae5"},
                        {"range":[1.3,1.4],"color":"#fef3c7"},
                        {"range":[1.4,2.0],"color":"#fee2e2"}]},
        title={"text":label,"font":{"size":12}}))
    fig.update_layout(height=190,margin=dict(t=30,b=5,l=15,r=15),paper_bgcolor="rgba(0,0,0,0)")
    return fig

def le8_radar(d):
    cats=list(d.keys()); vals=list(d.values())
    cats.append(cats[0]); vals.append(vals[0])
    fig=go.Figure(go.Scatterpolar(r=vals,theta=cats,fill="toself",
                                   fillcolor="rgba(59,130,246,.15)",
                                   line=dict(color="#3b82f6",width=2)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])),
                      height=290,margin=dict(t=15,b=15,l=25,r=25),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def egri_grafigi(df: pd.DataFrame) -> go.Figure:
    if df.empty: return go.Figure()
    fig = make_subplots(rows=3, cols=2, shared_xaxes=False,
                        subplot_titles=["LE8 Skoru","SCORE2 Risk %",
                                        "ABI (Sağ)","IAD (mmHg)",
                                        "Sistolik KB","VKİ"],
                        vertical_spacing=0.12, horizontal_spacing=0.1)
    tarihler = df["Tarih"].tolist()
    pairs = [("LE8",1,1,"#3b82f6"),("SCORE2%",1,2,"#ef4444"),
             ("ABI Sağ",2,1,"#8b5cf6"),("IAD (mmHg)",2,2,"#f59e0b"),
             ("Sistolik KB",3,1,"#dc2626"),("VKİ",3,2,"#64748b")]
    for col,r,c,color in pairs:
        if col not in df.columns: continue
        fig.add_trace(go.Scatter(x=tarihler,y=df[col].tolist(),mode="lines+markers",
                                  line=dict(color=color,width=2.5),
                                  marker=dict(size=7,color=color),
                                  name=col,showlegend=False), row=r, col=c)
    fig.update_layout(height=560,margin=dict(t=40,b=30,l=40,r=20),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    for ann in fig.layout.annotations: ann.font.size = 11
    return fig

def degisim_grafigi(rapor: dict) -> go.Figure:
    if not rapor: return go.Figure()
    satirlar, deltalar, renkler = [], [], []
    for g in rapor.get("iyiye",[]): satirlar.append(g["metrik"]); deltalar.append(g["delta"]); renkler.append("#16a34a")
    for g in rapor.get("kotye",[]): satirlar.append(g["metrik"]); deltalar.append(g["delta"]); renkler.append("#dc2626")
    for g in rapor.get("stabil",[]): satirlar.append(g["metrik"]); deltalar.append(0); renkler.append("#9ca3af")
    if not satirlar: return go.Figure()
    fig = go.Figure(go.Bar(x=deltalar,y=satirlar,orientation="h",marker_color=renkler,opacity=.8))
    fig.add_vline(x=0,line_color="#374151",line_dash="dash")
    fig.update_layout(title="Başlangıçtan Bu Yana Değişim",xaxis_title="Delta (pozitif = artış)",
                      height=320,margin=dict(t=40,b=30,l=10,r=20),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "hastalar"     not in st.session_state: st.session_state.hastalar  = _demo_hastalar()
if "aktif_idx"    not in st.session_state: st.session_state.aktif_idx = 0
if "hasta_gecmis" not in st.session_state:
    st.session_state.hasta_gecmis = {
        h.hasta_id: _demo_gecmis(h.hasta_id)
        for h in st.session_state.hastalar
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ❤️ KDS Platformu v3")
    st.caption("ABI/IAD — Hasta bazlı · Sağlık Eğrisi · Takip")

    st.markdown("**Hasta Seçimi**")
    isimler = [f"{h.ad} {h.soyad}" for h in st.session_state.hastalar]
    secim   = st.selectbox("Hasta", isimler, index=st.session_state.aktif_idx,
                            label_visibility="collapsed")
    st.session_state.aktif_idx = isimler.index(secim)
    h      = st.session_state.hastalar[st.session_state.aktif_idx]

    vki_s  = VKIHesaplayici.hesapla(h.boy_cm, h.kilo_kg)
    s2_s   = SCORE2Hesaplayici.hesapla(h)
    le8_s  = LE8Hesaplayici.hesapla(h)
    abi_s  = HastaABIHesaplayici.hesapla(h)
    rs     = SCORE2Hesaplayici.siniflandir(s2_s)

    st.markdown(f"""
    <div class="kds-metric">
      <div class="label">{h.yas} yaş · {'Erkek' if h.cinsiyet=='E' else 'Kadın'} · {h.hasta_id}</div>
      <div class="value" style="color:{vki_rengi(vki_s)}">{vki_s}</div>
      <div class="sub">VKİ · {VKIHesaplayici.siniflandir(vki_s)}</div>
    </div>
    <div class="kds-metric">
      <div class="label">LE8 · SCORE2</div>
      <div class="value" style="color:{le8_rengi(le8_s['toplam'])}">{le8_s['toplam']}</div>
      <div class="sub">SCORE2 %{s2_s} · {rs}</div>
    </div>
    <div class="kds-metric">
      <div class="label">ABI Sağ · IAD</div>
      <div class="value" style="color:{abi_rengi(abi_s['abi_sag'])}">{abi_s['abi_sag']:.3f}</div>
      <div class="sub">{abi_s['abi_sinif']} · IAD {abi_s['iad']:.1f} mmHg</div>
    </div>
    """, unsafe_allow_html=True)

    # Geçmiş kayıt sayısı
    gecmis = st.session_state.hasta_gecmis.get(h.hasta_id, [])
    st.caption(f"📁 {len(gecmis)} geçmiş kayıt")

    st.markdown("**Navigasyon**")
    sayfa = st.radio("Sayfa", [
        "🏠 Genel Bakış",
        "🔬 ABI/IAD Analizi",
        "⚖️ Çapraz Karşılaştırma",
        "📈 Sağlık Eğrisi & Takip",
        "💗 Life's Essential 8",
        "🧬 Cinsiyet Riski",
        "👥 Tüm Hastalar",
        "➕ Yeni Hasta",
    ], label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
# SAYFA: GENEL BAKIŞ
# ══════════════════════════════════════════════════════════════════════════════
if sayfa == "🏠 Genel Bakış":
    h      = st.session_state.hastalar[st.session_state.aktif_idx]
    gecmis = st.session_state.hasta_gecmis.get(h.hasta_id, [])
    kds    = tam_degerlendirme(h)
    abi    = HastaABIHesaplayici.hesapla(h)
    capraz = CaprazKarsilastirmaMotoru.karsilastir(kds, abi)

    st.title("🏠 KDS Genel Bakış")
    st.markdown(hasta_baslik_html(h, gecmis), unsafe_allow_html=True)

    if gecmis:
        with st.expander(f"📋 Test Geçmişi — {len(gecmis)} kayıt", expanded=False):
            st.markdown(test_tarihleri_html(gecmis), unsafe_allow_html=True)

    # Özet metrikler
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("VKİ",          f"{kds['vki']['deger']}",               kds["vki"]["sinif"])
    m2.metric("LE8",          f"{kds['le8']['toplam']}/100")
    m3.metric("SCORE2",       f"%{kds['score2']['yuzde']}",            kds["score2"]["sinif"])
    m4.metric("ABI (Sağ)",    f"{abi['abi_sag']:.3f}",                 abi["abi_sinif"])
    m5.metric("Birleşik Risk",capraz["combined_sinif"],
              "⚠ Uyumsuz" if not capraz["ayni_yon"] else "✓ Uyumlu")

    # Uyum banner
    if capraz["ayni_yon"] and capraz["kds_seviye"] >= 1:
        st.error(f"🚨 **YÜKSEK RİSK UYUMU:** KDS ({capraz['kds_risk']}) ve ABI ({capraz['abi_risk']}) "
                 f"testleri aynı yönde risk gösteriyor. Şiddetli önlem gerekli — aşağıyı inceleyin.")
    elif capraz["ayni_yon"] and capraz["kds_seviye"] == 0:
        st.success(f"✅ **DÜŞÜK RİSK UYUMU:** Her iki test de düşük risk onaylıyor.")
    else:
        st.warning(f"⚠️ **TEST UYUMSUZLUĞU:** KDS ({capraz['kds_risk']}) ile ABI ({capraz['abi_risk']}) "
                   f"farklı yönler gösteriyor. Dikkatli değerlendirme gerekiyor.")

    st.divider()
    c1,c2,c3 = st.columns(3)
    with c1:
        st.subheader("SCORE2 Riski")
        st.plotly_chart(score2_gauge(kds["score2"]["yuzde"]), use_container_width=True)
    with c2:
        st.subheader("ABI Sağ")
        st.plotly_chart(abi_gauge(abi["abi_sag"], "ABI Sağ"), use_container_width=True)
    with c3:
        st.subheader("ABI Sol")
        st.plotly_chart(abi_gauge(abi["abi_sol"], "ABI Sol"), use_container_width=True)

    # Öncelikli 3 tavsiye
    st.divider()
    st.subheader(f"Öncelikli Tavsiyeler ({len(capraz['tavsiyeler'])})")
    for tv in capraz["tavsiyeler"][:3]:
        st.markdown(tavsiye_html(tv), unsafe_allow_html=True)
    if len(capraz["tavsiyeler"]) > 3:
        st.caption(f"➕ {len(capraz['tavsiyeler'])-3} ek tavsiye için **⚖️ Çapraz Karşılaştırma** sayfasını açın.")

    st.divider()
    sozluk_goster("🏠 Genel Bakış")


# ══════════════════════════════════════════════════════════════════════════════
# SAYFA: ABI/IAD ANALİZİ (HASTA BAZLI)
# ══════════════════════════════════════════════════════════════════════════════
elif sayfa == "🔬 ABI/IAD Analizi":
    h      = st.session_state.hastalar[st.session_state.aktif_idx]
    gecmis = st.session_state.hasta_gecmis.get(h.hasta_id, [])
    st.title("🔬 ABI / IAD Analizi")
    st.markdown(hasta_baslik_html(h, gecmis), unsafe_allow_html=True)
    if gecmis:
        with st.expander(f"📋 Test Geçmişi — {len(gecmis)} kayıt"):
            st.markdown(test_tarihleri_html(gecmis), unsafe_allow_html=True)
    st.caption("Veriler hasta kaydındaki tansiyon ölçümlerinden otomatik hesaplanır. "
               "Güncelleme için aşağıdaki formu kullanın.")

    # ── Ölçüm Güncelleme Formu ──────────────────────────────────
    with st.expander("✏️ Tansiyon Ölçümlerini Güncelle", expanded=False):
        with st.form("abi_form"):
            st.caption("Tüm değerler mmHg cinsinden")
            r1c1,r1c2 = st.columns(2)
            sag_kol  = r1c1.number_input("Sağ Kol SBP",  50,250,int(h.sbp_sag_kol))
            sol_kol  = r1c2.number_input("Sol Kol SBP",  50,250,int(h.sbp_sol_kol))
            r2c1,r2c2 = st.columns(2)
            sag_ayak = r2c1.number_input("Sağ Ayak SBP", 50,300,int(h.sbp_sag_ayak))
            sol_ayak = r2c2.number_input("Sol Ayak SBP", 50,300,int(h.sbp_sol_ayak))
            guncelle = st.form_submit_button("Kaydet & Hesapla")
        if guncelle:
            h.sbp_sag_kol  = float(sag_kol)
            h.sbp_sol_kol  = float(sol_kol)
            h.sbp_sag_ayak = float(sag_ayak)
            h.sbp_sol_ayak = float(sol_ayak)
            st.success("✅ Ölçümler güncellendi.")

    abi = HastaABIHesaplayici.hesapla(h)

    # ── Metrikler ────────────────────────────────────────────────
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("ABI Sağ",    f"{abi['abi_sag']:.3f}",  abi["abi_sinif"])
    m2.metric("ABI Sol",    f"{abi['abi_sol']:.3f}")
    m3.metric("IAD",        f"{abi['iad']:.1f} mmHg",  abi["iad_sinif"])
    m4.metric("Asimetri",   f"{abs(abi['abi_sag']-abi['abi_sol']):.3f}",
              "Dikkat" if abs(abi['abi_sag']-abi['abi_sol']) > 0.15 else "Normal")

    st.divider()
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(abi_gauge(abi["abi_sag"], "ABI Sağ"), use_container_width=True)
        st.plotly_chart(abi_gauge(abi["abi_sol"], "ABI Sol"), use_container_width=True)
    with c2:
        # ABI eşik şeması
        fig = go.Figure()
        zones = [(0.0,0.5,"#991b1b","Kritik PAH"),
                 (0.5,0.7,"#dc2626","Şiddetli PAH"),
                 (0.7,0.9,"#f97316","Ilımlı PAH"),
                 (0.9,1.0,"#fbbf24","Sınırda"),
                 (1.0,1.3,"#16a34a","Normal"),
                 (1.3,1.4,"#fbbf24","Sınırda Yüksek"),
                 (1.4,2.0,"#dc2626","Kalsif. Riski")]
        for lo,hi,col,lbl in zones:
            fig.add_shape(type="rect", x0=lo,x1=hi,y0=0,y1=1,
                          fillcolor=col,opacity=.2,line_width=0)
            fig.add_annotation(x=(lo+hi)/2, y=0.5, text=lbl,
                                showarrow=False, font=dict(size=9,color=col))
        for val, lbl, col in [(abi["abi_sag"],"Sağ","#1e40af"),(abi["abi_sol"],"Sol","#7c3aed")]:
            fig.add_vline(x=val, line_color=col, line_dash="dash", line_width=2.5,
                          annotation_text=f"{lbl}: {val:.3f}",
                          annotation_font_size=11, annotation_font_color=col)
        fig.update_layout(title="ABI Bölge Şeması", height=180,
                          xaxis=dict(range=[0,1.8], title="ABI Değeri"),
                          yaxis=dict(visible=False),
                          margin=dict(t=40,b=30,l=10,r=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # IAD göstergesi
        iad_col = "#16a34a" if abi["iad"] < 10 else "#f59e0b" if abi["iad"] < 20 else "#dc2626"
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number", value=abi["iad"],
            number={"suffix":" mmHg","font":{"size":18}},
            gauge={"axis":{"range":[0,30]},
                   "bar":{"color":iad_col,"thickness":.3},
                   "steps":[{"range":[0,10],"color":"#d1fae5"},
                             {"range":[10,20],"color":"#fef3c7"},
                             {"range":[20,30],"color":"#fee2e2"}]},
            title={"text":"IAD","font":{"size":12}}))
        fig2.update_layout(height=180,margin=dict(t=30,b=5,l=15,r=15),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Ölçüm tablosu ────────────────────────────────────────────
    st.divider()
    st.subheader("Ölçüm Detayları")
    olcum_df = pd.DataFrame([
        {"Ölçüm Noktası":"Sağ Kol SBP","Değer (mmHg)":h.sbp_sag_kol},
        {"Ölçüm Noktası":"Sol Kol SBP","Değer (mmHg)":h.sbp_sol_kol},
        {"Ölçüm Noktası":"Sağ Ayak SBP","Değer (mmHg)":h.sbp_sag_ayak},
        {"Ölçüm Noktası":"Sol Ayak SBP","Değer (mmHg)":h.sbp_sol_ayak},
        {"Ölçüm Noktası":"ABI Sağ (hesap)","Değer (mmHg)":f"{abi['abi_sag']:.3f}"},
        {"Ölçüm Noktası":"ABI Sol (hesap)","Değer (mmHg)":f"{abi['abi_sol']:.3f}"},
        {"Ölçüm Noktası":"IAD","Değer (mmHg)":f"{abi['iad']:.1f}"},
    ])
    st.dataframe(olcum_df, hide_index=True, use_container_width=True)

    # ── Kaydet butonu ────────────────────────────────────────────
    st.divider()
    not_txt = st.text_input("Oturum notu (opsiyonel)", placeholder="Ölçüm koşulları, gözlemler...")
    if st.button("📥 Bu Testi Takip Dosyasına Kaydet", type="primary"):
        kds = tam_degerlendirme(h)
        kayit = SaglikEgrisiMotoru.kayit_olustur(h, kds, abi, not_txt)
        if h.hasta_id not in st.session_state.hasta_gecmis:
            st.session_state.hasta_gecmis[h.hasta_id] = []
        st.session_state.hasta_gecmis[h.hasta_id].append(kayit)
        st.success(f"✅ Kayıt eklendi. Toplam {len(st.session_state.hasta_gecmis[h.hasta_id])} kayıt.")

    st.divider()
    sozluk_goster("🔬 ABI/IAD Analizi")


# ══════════════════════════════════════════════════════════════════════════════
# SAYFA: ÇAPRAZ KARŞILAŞTIRMA
# ══════════════════════════════════════════════════════════════════════════════
elif sayfa == "⚖️ Çapraz Karşılaştırma":
    h      = st.session_state.hastalar[st.session_state.aktif_idx]
    gecmis = st.session_state.hasta_gecmis.get(h.hasta_id, [])
    kds    = tam_degerlendirme(h)
    abi    = HastaABIHesaplayici.hesapla(h)
    capraz = CaprazKarsilastirmaMotoru.karsilastir(kds, abi)

    st.title("⚖️ KDS × ABI/IAD Çapraz Karşılaştırma")
    st.markdown(hasta_baslik_html(h, gecmis), unsafe_allow_html=True)
    if gecmis:
        with st.expander(f"📋 Test Geçmişi — {len(gecmis)} kayıt"):
            st.markdown(test_tarihleri_html(gecmis), unsafe_allow_html=True)

    # ── Uyum durumu banner ───────────────────────────────────────
    ayni = capraz["ayni_yon"]
    ks   = capraz["kds_seviye"]
    if ayni and ks >= 1:
        st.error("🚨 **TEST UYUMU: İKİ TEST DE YÜKSEK RİSK GÖSTERİYOR**  "
                 "Aşağıdaki şiddetli tavsiyeler klinisyen değerlendirmesiyle uygulanmalıdır.")
    elif ayni and ks == 0:
        st.success("✅ **TEST UYUMU: İKİ TEST DE DÜŞÜK RİSK ONAYLIYOR**")
    else:
        st.warning("⚠️ **TEST UYUMSUZLUĞU:** Testler farklı risk yönleri gösteriyor. "
                   "Gerekçeli takip kararları aşağıda açıklanmıştır.")

    # ── Yan yana test özeti ──────────────────────────────────────
    st.divider()
    st.subheader("Test Sonuçları Karşılaştırması")
    cc1, cc2, cc3 = st.columns([2,2,1])

    with cc1:
        st.markdown("##### 🫀 KDS (SCORE2) Sonucu")
        st.plotly_chart(score2_gauge(kds["score2"]["yuzde"]), use_container_width=True)
        kd1,kd2 = st.columns(2)
        kd1.metric("Risk Sınıfı",  kds["score2"]["sinif"])
        kd2.metric("LE8 Skoru",    f"{kds['le8']['toplam']}/100")
        kd1.metric("CKM Evresi",   f"Evre {kds['ckm_evresi']}")
        kd2.metric("VKİ",          f"{kds['vki']['deger']}  ({kds['vki']['sinif']})")

    with cc2:
        st.markdown("##### 🩺 ABI/IAD Sonucu")
        st.plotly_chart(abi_gauge(abi["abi_sag"]), use_container_width=True)
        ad1,ad2 = st.columns(2)
        ad1.metric("ABI Sınıfı",   abi["abi_sinif"])
        ad2.metric("IAD Sınıfı",   abi["iad_sinif"])
        ad1.metric("ABI Sağ",      f"{abi['abi_sag']:.3f}")
        ad2.metric("IAD",          f"{abi['iad']:.1f} mmHg")

    with cc3:
        st.markdown("##### 🔗 Uyum")
        st.markdown(f"""
        <div style="text-align:center;padding:1rem;border-radius:12px;
             background:{'#fef2f2' if (ayni and ks>=1) else '#f0fdf4' if (ayni and ks==0) else '#fffbeb'};
             margin-top:.5rem">
            <div style="font-size:2rem">{'🚨' if (ayni and ks>=1) else '✅' if (ayni and ks==0) else '⚠️'}</div>
            <div style="font-weight:600;font-size:.85rem;margin-top:.3rem">
                {'UYUMLU<br>YÜK. RİSK' if (ayni and ks>=1)
                 else 'UYUMLU<br>DÜŞÜK RİSK' if (ayni and ks==0)
                 else 'UYUMSUZ<br>DİKKAT'}
            </div>
            <div style="font-size:.72rem;color:#6b7280;margin-top:.3rem">
                KDS: {capraz['kds_risk']}<br>ABI: {capraz['abi_risk']}
            </div>
            <div style="font-weight:600;margin-top:.5rem;font-size:.9rem;
                 color:{'#dc2626' if capraz['combined_sinif'] in ('Yüksek','Çok Yüksek')
                        else '#d97706' if capraz['combined_sinif']=='Orta' else '#16a34a'}">
                Birleşik:<br>{capraz['combined_sinif']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── LE8 radar ───────────────────────────────────────────────
    st.divider()
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("LE8 Profili")
        st.plotly_chart(le8_radar(kds["le8"]["detay"]), use_container_width=True)

    with c_right:
        # KDS vs ABI risk bileşenlerini spider'da karşılaştır
        st.subheader("Risk Bileşen Karşılaştırması")
        categories = ["KB Riski","Lipid Riski","Glisemi Riski",
                      "Kilo Riski","Sigara Riski","ABI Riski","IAD Riski"]
        kds_vals = [
            min(100, max(0, (kds["hasta"].sistolik_kb - 110) * 2)),
            min(100, max(0, (kds["hasta"].total_kolesterol - 150) * 0.5)),
            min(100, max(0, (kds["hasta"].hba1c - 4.5) * 20)),
            min(100, max(0, (kds["vki"]["deger"] - 18) * 4)),
            0 if h.sigara=="hayir" else 50 if h.sigara=="birakti" else 100,
            0, 0,
        ]
        abi_vals = [
            0, 0, 0, 0, 0,
            min(100, max(0, (1.2 - abi["abi_sag"]) * 200)) if abi["abi_sag"] < 1.2 else 0,
            min(100, abi["iad"] * 4),
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=kds_vals+[kds_vals[0]], theta=categories+[categories[0]],
            fill="toself", fillcolor="rgba(59,130,246,.15)",
            line=dict(color="#3b82f6",width=2), name="KDS"))
        fig.add_trace(go.Scatterpolar(
            r=abi_vals+[abi_vals[0]], theta=categories+[categories[0]],
            fill="toself", fillcolor="rgba(220,38,38,.15)",
            line=dict(color="#dc2626",width=2), name="ABI/IAD"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])),
                          height=290, margin=dict(t=15,b=30,l=25,r=25),
                          paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h",y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    # ── TAVSİYELER ───────────────────────────────────────────────
    st.divider()
    if ayni and ks >= 1:
        st.subheader("🚨 ŞİDDETLİ TAVSİYELER — Her İki Test Yüksek Risk Onaylıyor")
    elif not ayni:
        st.subheader("⚠️ TEĞİT TAKİP KARARLARI — Test Uyumsuzluğu Durumunda")
    else:
        st.subheader("✅ Koruyucu Öneriler")

    for tv in capraz["tavsiyeler"]:
        st.markdown(tavsiye_html(tv), unsafe_allow_html=True)

    # ── Rapor indirme ────────────────────────────────────────────
    st.divider()
    rapor_md = f"""# KDS × ABI/IAD Çapraz Karşılaştırma Raporu
**Hasta:** {h.ad} {h.soyad} ({h.hasta_id})  
**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Yaş/Cinsiyet:** {h.yas} / {'Erkek' if h.cinsiyet=='E' else 'Kadın'}

## Test Sonuçları

| Test | Değer | Sınıf | Risk |
|------|-------|-------|------|
| SCORE2 (KDS) | %{kds['score2']['yuzde']} | — | {kds['score2']['sinif']} |
| LE8 Skoru | {kds['le8']['toplam']}/100 | — | — |
| CKM Evresi | Evre {kds['ckm_evresi']} | — | — |
| ABI Sağ | {abi['abi_sag']:.3f} | {abi['abi_sinif']} | {capraz['abi_risk']} |
| ABI Sol | {abi['abi_sol']:.3f} | — | — |
| IAD | {abi['iad']:.1f} mmHg | {abi['iad_sinif']} | — |

## Uyum Durumu

- **Test Uyumu:** {'UYUMLU' if capraz['ayni_yon'] else 'UYUMSUZ'}
- **KDS Risk:** {capraz['kds_risk']}  
- **ABI Risk:** {capraz['abi_risk']}  
- **Birleşik Risk:** {capraz['combined_sinif']}

## Tavsiyeler

"""
    for tv in capraz["tavsiyeler"]:
        rapor_md += f"""### [{tv['tip']}] {tv['baslik']}
{tv['mesaj']}

**Eylem:** {tv['eylem']}

"""
    rapor_md += "\n> ⚕️ Bu rapor bilgilendirme amaçlıdır. Klinik kararlar için hekim değerlendirmesi zorunludur."

    st.download_button("⬇ Çapraz Karşılaştırma Raporu (.md)",
                       data=rapor_md.encode("utf-8"),
                       file_name=f"capraz_{h.hasta_id}_{datetime.now().strftime('%Y%m%d')}.md",
                       mime="text/markdown")
    st.divider()
    sozluk_goster("⚖️ Çapraz Karşılaştırma")


# ══════════════════════════════════════════════════════════════════════════════
# SAYFA: SAĞLIK EĞRİSİ & TAKİP
# ══════════════════════════════════════════════════════════════════════════════
elif sayfa == "📈 Sağlık Eğrisi & Takip":
    h      = st.session_state.hastalar[st.session_state.aktif_idx]
    gecmis = st.session_state.hasta_gecmis.get(h.hasta_id, [])

    st.title("📈 Sağlık Eğrisi & Takip")
    st.markdown(hasta_baslik_html(h, gecmis), unsafe_allow_html=True)

    # Tüm test tarihleri — bu sayfada her zaman açık
    st.subheader("Test Tarihleri & Geçen Süreler")
    st.markdown(test_tarihleri_html(gecmis), unsafe_allow_html=True)

    if len(gecmis) < 2:
        st.info("Sağlık eğrisi için en az 2 kayıt gereklidir. "
                "**🔬 ABI/IAD Analizi** sayfasından yeni test kaydı ekleyin.")
        if gecmis:
            k = gecmis[0]
            st.caption(f"Mevcut tek kayıt: {_tarih_goster(k.tarih)} ({_gun_farki_str(k.tarih)})")
            st.json({"Tarih":k.tarih,"LE8":k.kds_le8_toplam,"SCORE2":k.kds_score2,
                     "ABI Sağ":k.abi_sag,"IAD":k.iad_mmhg})
    else:
        df_egri = SaglikEgrisiMotoru.egri_dataframe(gecmis)
        degisim = SaglikEgrisiMotoru.degisim_raporu(gecmis)

        # ── Özet değişim kartları ────────────────────────────────
        iyiler = degisim.get("iyiye",[])
        kotuler= degisim.get("kotye",[])
        stabils= degisim.get("stabil",[])

        col_i,col_k,col_s = st.columns(3)
        col_i.metric("İyileşen Metrik", len(iyiler), "↑", delta_color="normal")
        col_k.metric("Kötüleşen Metrik",len(kotuler),"↓", delta_color="inverse")
        col_s.metric("Stabil Metrik",   len(stabils))

        st.divider()
        st.subheader("Sağlık Eğrisi (Zaman İçi Değişim)")
        st.plotly_chart(egri_grafigi(df_egri), use_container_width=True)

        # ── Değişim özet grafiği ─────────────────────────────────
        st.divider()
        st.subheader("Başlangıçtan Bugüne Değişim Özeti")
        col_gr, col_tb = st.columns([1,1])
        with col_gr:
            st.plotly_chart(degisim_grafigi(degisim), use_container_width=True)
        with col_tb:
            # İyiye/kötüye tablo
            st.markdown("**İyileşen Metrikler**")
            for m in iyiler:
                st.markdown(
                    f'<span class="iyi-badge">↑ {m["metrik"]}</span> '
                    f'{m["baslangic"]} → **{m["son"]}** (Δ {m["delta"]:+.2f})',
                    unsafe_allow_html=True)
            st.markdown("**Kötüleşen Metrikler**")
            for m in kotuler:
                st.markdown(
                    f'<span class="kotu-badge">↓ {m["metrik"]}</span> '
                    f'{m["baslangic"]} → **{m["son"]}** (Δ {m["delta"]:+.2f})',
                    unsafe_allow_html=True)
            if stabils:
                st.markdown("**Stabil Metrikler**")
                for m in stabils:
                    st.markdown(
                        f'<span class="stab-badge">→ {m["metrik"]}</span> '
                        f'{m["son"]}',
                        unsafe_allow_html=True)

        # ── Geçmiş kayıt tablosu ─────────────────────────────────
        st.divider()
        st.subheader("Tüm Test Kayıtları")
        st.dataframe(df_egri.drop(columns=["Notlar"],errors="ignore"),
                     hide_index=True, use_container_width=True,
                     column_config={
                         "LE8":       st.column_config.ProgressColumn("LE8",min_value=0,max_value=100),
                         "SCORE2%":   st.column_config.NumberColumn("SCORE2%",format="%.1f"),
                         "ABI Sağ":   st.column_config.NumberColumn("ABI",format="%.3f"),
                     })

        # ── Takip raporu ─────────────────────────────────────────
        st.divider()
        st.subheader("Takip Raporu")
        rapor_md = f"""# Sağlık Takip Raporu — {h.ad} {h.soyad}
**Hasta ID:** {h.hasta_id}  
**Oluşturma:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Takip Aralığı:** {degisim['ilk_tarih']} → {degisim['son_tarih']}  
**Toplam Kayıt:** {degisim['toplam_kayit']}

## İyileşen Metrikler (✅)

"""
        for m in iyiler:
            rapor_md += f"- **{m['metrik']}**: {m['baslangic']} → {m['son']} (Δ {m['delta']:+.3f})\n"
        rapor_md += "\n## Kötüleşen Metrikler (⚠️)\n\n"
        for m in kotuler:
            rapor_md += f"- **{m['metrik']}**: {m['baslangic']} → {m['son']} (Δ {m['delta']:+.3f})\n"
        rapor_md += "\n## Stabil Metrikler (→)\n\n"
        for m in stabils:
            rapor_md += f"- **{m['metrik']}**: {m['son']}\n"

        rapor_md += f"""
## Klinik Özet

| Metrik | İlk Değer | Son Değer | Yön |
|--------|-----------|-----------|-----|
| LE8 | {gecmis[0].kds_le8_toplam} | {gecmis[-1].kds_le8_toplam} | {'↑' if gecmis[-1].kds_le8_toplam > gecmis[0].kds_le8_toplam else '↓'} |
| SCORE2% | {gecmis[0].kds_score2} | {gecmis[-1].kds_score2} | {'↑' if gecmis[-1].kds_score2 > gecmis[0].kds_score2 else '↓'} |
| ABI Sağ | {gecmis[0].abi_sag:.3f} | {gecmis[-1].abi_sag:.3f} | {'↑' if gecmis[-1].abi_sag > gecmis[0].abi_sag else '↓'} |
| IAD | {gecmis[0].iad_mmhg:.1f} | {gecmis[-1].iad_mmhg:.1f} | {'↑' if gecmis[-1].iad_mmhg > gecmis[0].iad_mmhg else '↓'} |
| Kilo | {gecmis[0].kilo_kg} | {gecmis[-1].kilo_kg} | {'↑' if gecmis[-1].kilo_kg > gecmis[0].kilo_kg else '↓'} |
| Sistolik KB | {gecmis[0].sistolik_kb} | {gecmis[-1].sistolik_kb} | {'↑' if gecmis[-1].sistolik_kb > gecmis[0].sistolik_kb else '↓'} |

> ⚕️ Bilgilendirme amaçlıdır. Klinik kararlar için hekim değerlendirmesi zorunludur.
"""
        with st.expander("📄 Tam Raporu Görüntüle"):
            st.markdown(rapor_md)
        st.download_button("⬇ Takip Raporu İndir (.md)",
                           data=rapor_md.encode("utf-8"),
                           file_name=f"takip_{h.hasta_id}_{datetime.now().strftime('%Y%m%d')}.md",
                           mime="text/markdown")
        st.divider()
        sozluk_goster("📈 Sağlık Eğrisi & Takip")


# ══════════════════════════════════════════════════════════════════════════════
# SAYFA: LIFE'S ESSENTIAL 8
# ══════════════════════════════════════════════════════════════════════════════
elif sayfa == "💗 Life's Essential 8":
    h      = st.session_state.hastalar[st.session_state.aktif_idx]
    gecmis = st.session_state.hasta_gecmis.get(h.hasta_id, [])
    le8    = LE8Hesaplayici.hesapla(h)
    st.title("💗 AHA Life's Essential 8")
    st.markdown(hasta_baslik_html(h, gecmis), unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(le8_radar(le8["detay"]), use_container_width=True)
    with c2:
        st.metric("Toplam LE8",f"{le8['toplam']}/100"); st.divider()
        for m,s in le8["detay"].items():
            mc,bc,vc = st.columns([3,5,1])
            mc.caption(m); bc.progress(s/100); vc.markdown(f"**{s}**")
    st.divider()
    sozluk_goster("💗 Life's Essential 8")


# ══════════════════════════════════════════════════════════════════════════════
# SAYFA: CİNSİYET RİSKİ
# ══════════════════════════════════════════════════════════════════════════════
elif sayfa == "🧬 Cinsiyet Riski":
    h      = st.session_state.hastalar[st.session_state.aktif_idx]
    gecmis = st.session_state.hasta_gecmis.get(h.hasta_id, [])
    st.title("🧬 Cinsiyete Özgü KDS Risk")
    st.markdown(hasta_baslik_html(h, gecmis), unsafe_allow_html=True)
    te,tk = st.tabs(["🔵 Erkek","🩷 Kadın"])
    with te:
        st.info("Erkeklerde KVH olayları **10 yıl daha erken** görülür.")
        st.dataframe(pd.DataFrame({
            "Parametre":["Bel eşiği 1","Bel eşiği 2 (Yüksek)","B/K oranı","KVH yaşı","Obez→Diyabet","Obez→HT"],
            "Değer":[">94 cm",">102 cm",">0.90","55–65","3.5×","2.9×"],
        }),hide_index=True,use_container_width=True)
    with tk:
        st.warning("Menopoz sonrası östrojen kaybıyla KDS riski **hızla artar**.")
        st.dataframe(pd.DataFrame({
            "Parametre":["Bel eşiği 1","Bel eşiği 2","B/K oranı","Menopoz sonrası","Gest. diyabet→T2DM","PCOS"],
            "Değer":[">80 cm",">88 cm",">0.85","+40%","+7×","↑ insülin direnci"],
        }),hide_index=True,use_container_width=True)
        k1,k2,k3,k4 = st.columns(4)
        k1.info("🤰 **Gebelik komp.**\nUzun vadeli KDS riskini artırır")
        k2.warning("🔄 **PCOS**\nİnsülin direnci + obezite")
        k3.warning("📉 **Menopoz sonrası**\nHDL↓ LDL↑ TG↑")
        k4.error("⚠️ **Otoimmün**\nErken ateroskleroz")
    st.divider()
    sozluk_goster("🧬 Cinsiyet Riski")


# ══════════════════════════════════════════════════════════════════════════════
# SAYFA: TÜM HASTALAR
# ══════════════════════════════════════════════════════════════════════════════
elif sayfa == "👥 Tüm Hastalar":
    st.title("👥 Hasta Listesi")
    st.caption(f"Toplam {len(st.session_state.hastalar)} kayıtlı hasta")
    rows = []
    for h in st.session_state.hastalar:
        v   = VKIHesaplayici.hesapla(h.boy_cm,h.kilo_kg)
        s2  = SCORE2Hesaplayici.hesapla(h)
        le8 = LE8Hesaplayici.hesapla(h)
        abi = HastaABIHesaplayici.hesapla(h)
        gc  = len(st.session_state.hasta_gecmis.get(h.hasta_id,[]))
        rows.append({
            "Ad Soyad":f"{h.ad} {h.soyad}","ID":h.hasta_id,"Yaş":h.yas,"C":h.cinsiyet,
            "VKİ":v,"VKİ Sınıfı":VKIHesaplayici.siniflandir(v),
            "LE8":le8["toplam"],"SCORE2%":s2,"Risk":SCORE2Hesaplayici.siniflandir(s2),
            "ABI Sağ":round(abi["abi_sag"],3),"ABI Sınıfı":abi["abi_sinif"],
            "IAD":round(abi["iad"],1),
            "Kayıt":gc,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={
                     "LE8":    st.column_config.ProgressColumn("LE8",min_value=0,max_value=100),
                     "SCORE2%":st.column_config.NumberColumn("SCORE2%",format="%.1f%%"),
                     "VKİ":    st.column_config.NumberColumn("VKİ",format="%.1f"),
                     "ABI Sağ":st.column_config.NumberColumn("ABI",format="%.3f"),
                 })
    c1,c2 = st.columns(2)
    with c1:
        fig=px.scatter(df, x="SCORE2%", y="ABI Sağ", color="Risk", size="VKİ",
                       hover_data=["Ad Soyad"],
                       title="SCORE2% × ABI Sağ — Tüm Hastalar",
                       color_discrete_map={"Düşük":"#16a34a","Orta":"#d97706",
                                            "Yüksek":"#dc2626","Çok Yüksek":"#7f1d1d"})
        fig.add_hline(y=0.9, line_dash="dash", line_color="crimson",
                      annotation_text="ABI 0.9 (PAH eşiği)")
        fig.update_layout(height=340,paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig=px.bar(df.sort_values("LE8"), x="Ad Soyad", y="LE8",
                   color="Risk", title="Hasta LE8 Skorları",
                   color_discrete_map={"Düşük":"#16a34a","Orta":"#d97706",
                                        "Yüksek":"#dc2626","Çok Yüksek":"#7f1d1d"})
        fig.update_layout(height=340,paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    sozluk_goster("👥 Tüm Hastalar")


# ══════════════════════════════════════════════════════════════════════════════
# SAYFA: YENİ HASTA
# ══════════════════════════════════════════════════════════════════════════════
elif sayfa == "➕ Yeni Hasta":
    st.title("➕ Yeni Hasta Kaydı")
    with st.form("yeni_hasta"):
        st.subheader("Demografik")
        c1,c2 = st.columns(2)
        ad_f=c1.text_input("Ad"); soyad_f=c2.text_input("Soyad")
        c1,c2,c3 = st.columns(3)
        yas_f=c1.number_input("Yaş",18,110,50)
        cins_f=c2.selectbox("Cinsiyet",["E","K"])
        diab_f=c3.selectbox("Diyabet",["yok","prediyabet","tip1","tip2"])

        st.subheader("Antropometrik")
        c1,c2,c3,c4 = st.columns(4)
        boy_f=c1.number_input("Boy (cm)",100,220,170)
        kilo_f=c2.number_input("Kilo (kg)",30,250,75)
        bel_f=c3.number_input("Bel (cm)",40,200,90)
        kalca_f=c4.number_input("Kalça (cm)",40,200,100)
        vki_p=VKIHesaplayici.hesapla(boy_f,kilo_f)
        st.info(f"📊 VKİ: **{vki_p}** — {VKIHesaplayici.siniflandir(vki_p)}")

        st.subheader("Klinik")
        c1,c2,c3,c4 = st.columns(4)
        sbp_f=c1.number_input("Sistolik KB",60,250,120)
        dbp_f=c2.number_input("Diyastolik KB",40,150,80)
        tc_f=c3.number_input("Total Kol",50,500,200)
        hdl_f=c4.number_input("HDL",10,150,50)
        c1,c2,c3,c4 = st.columns(4)
        ldl_f=c1.number_input("LDL",10,400,130)
        glu_f=c2.number_input("Açlık Glukoz",50,400,100)
        hba1c_f=c3.number_input("HbA1c (%)",3.0,15.0,5.5,0.1)
        trig_f=c4.number_input("Trigliserid",30,1000,150)

        st.subheader("ABI/IAD Tansiyon Ölçümleri")
        st.caption("Periferik arter değerlendirmesi için gerekli (mmHg)")
        ac1,ac2,ac3,ac4 = st.columns(4)
        sbp_sk=ac1.number_input("Sağ Kol SBP",50,250,125)
        sbp_slk=ac2.number_input("Sol Kol SBP",50,250,122)
        sbp_sa=ac3.number_input("Sağ Ayak SBP",50,300,130)
        sbp_sla=ac4.number_input("Sol Ayak SBP",50,300,128)
        # Anlık ABI önizleme
        if sbp_sk > 0 and sbp_slk > 0:
            abi_onizleme_sag = round(sbp_sa/sbp_sk, 3)
            abi_onizleme_sol = round(sbp_sla/sbp_slk, 3)
            iad_onizleme = abs(sbp_sk - sbp_slk)
            st.info(f"📐 ABI Önizleme: Sağ={abi_onizleme_sag:.3f}  Sol={abi_onizleme_sol:.3f}  IAD={iad_onizleme:.1f} mmHg")

        st.subheader("Yaşam Tarzı")
        c1,c2,c3 = st.columns(3)
        sig_f=c1.selectbox("Sigara",["hayir","evet","birakti"])
        akt_f=c2.number_input("Aktivite (dk/hafta)",0,1000,150)
        uyku_f=c3.number_input("Uyku (saat)",1.0,14.0,7.0,0.5)

        meno_f=pcos_f=geb_f=hrt_f=None
        if cins_f=="K":
            st.subheader("Kadına Özgü")
            c1,c2,c3,c4 = st.columns(4)
            meno_f=c1.selectbox("Menopoz",["pre","peri","post"])
            pcos_f=c2.checkbox("PCOS")
            geb_f=c3.selectbox("Gebelik Komp.",["yok","preeklampsi","gestasyonel_diyabet"])
            hrt_f=c4.checkbox("HRT")

        kaydet = st.form_submit_button("💾 Kaydet, Değerlendir & İlk Kaydı Oluştur", type="primary")

    if kaydet:
        if not ad_f or not soyad_f:
            st.error("Ad ve soyad zorunludur.")
        else:
            yeni = Hasta(
                ad=ad_f, soyad=soyad_f, yas=int(yas_f), cinsiyet=cins_f,
                boy_cm=float(boy_f), kilo_kg=float(kilo_f),
                bel_cm=float(bel_f), kalca_cm=float(kalca_f),
                sistolik_kb=int(sbp_f), diyastolik_kb=int(dbp_f),
                total_kolesterol=float(tc_f), hdl=float(hdl_f), ldl=float(ldl_f),
                aclik_kan_sekeri=float(glu_f), hba1c=float(hba1c_f), trigliserid=float(trig_f),
                sigara=sig_f, aktivite_dk_hafta=int(akt_f), uyku_saat=float(uyku_f),
                diyabet=diab_f, menopoz=meno_f, pcos=pcos_f or False,
                gebelik_komplikasyon=geb_f, hrt=hrt_f or False,
                sbp_sag_kol=float(sbp_sk), sbp_sol_kol=float(sbp_slk),
                sbp_sag_ayak=float(sbp_sa), sbp_sol_ayak=float(sbp_sla),
            )
            st.session_state.hastalar.append(yeni)
            st.session_state.aktif_idx = len(st.session_state.hastalar) - 1

            # İlk test kaydını otomatik oluştur
            kds_yeni = tam_degerlendirme(yeni)
            abi_yeni = HastaABIHesaplayici.hesapla(yeni)
            ilk_kayit = SaglikEgrisiMotoru.kayit_olustur(yeni, kds_yeni, abi_yeni, "İlk kayıt")
            st.session_state.hasta_gecmis[yeni.hasta_id] = [ilk_kayit]

            # Çapraz değerlendirme
            capraz_yeni = CaprazKarsilastirmaMotoru.karsilastir(kds_yeni, abi_yeni)

            st.success(f"✅ {yeni.ad} {yeni.soyad} kaydedildi ve ilk test kaydı oluşturuldu.")
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("VKİ",    kds_yeni["vki"]["deger"],         kds_yeni["vki"]["sinif"])
            m2.metric("LE8",    f"{kds_yeni['le8']['toplam']}/100")
            m3.metric("SCORE2", f"%{kds_yeni['score2']['yuzde']}", kds_yeni["score2"]["sinif"])
            m4.metric("ABI",    f"{abi_yeni['abi_sag']:.3f}",      abi_yeni["abi_sinif"])
            m5.metric("Birleşik Risk", capraz_yeni["combined_sinif"])

            st.subheader("Anlık Tavsiyeler")
            for tv in capraz_yeni["tavsiyeler"]:
                st.markdown(tavsiye_html(tv), unsafe_allow_html=True)

    st.divider()
    sozluk_goster("➕ Yeni Hasta")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption("⚕️ KDS Platformu v3 — bilgi desteği amaçlıdır. "
           "Klinik kararlar için hekim değerlendirmesi zorunludur. "
           "| LE8 (AHA) · SCORE2 · SELECT RCT (Lincoff 2023) · ABI/IAD")
