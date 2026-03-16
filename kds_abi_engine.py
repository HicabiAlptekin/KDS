#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KDS Platformu — ABI/IAD Analiz Motoru
========================================
Orijinal abi_analysis_test.py'deki ABIAnalyzer sınıfını
DEĞİŞTİRMEDEN miras alır.

Ek sorumluluklar:
  - Streamlit için sözlük tabanlı özet çıktısı
  - İki bağımsız çalışmanın karşılaştırma istatistikleri
  - Hem metin hem Markdown rapor üretimi
  - Plotly grafik üretimi (Streamlit uyumlu)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Orijinal sınıfı değiştirmeden import et
from abi_analysis_test import ABIAnalyzer


# ─────────────────────────────────────────────────────────────
# MOTOR SINIFI
# ─────────────────────────────────────────────────────────────

class KDSABIEngine(ABIAnalyzer):
    """
    ABIAnalyzer'ı BOZMADAN genişletir.
    Yalnızca yeni metotlar eklenir.
    """

    def __init__(self, study_label: str = "Çalışma", output_dir: str = "abi_outputs"):
        super().__init__(output_dir=output_dir)
        self.study_label = study_label
        self._summary_cache: dict | None = None

    # ── Özet sözlüğü (Streamlit metrik kartları için) ──────────
    def get_summary(self) -> dict:
        """Streamlit metrik kartları ve karşılaştırma için sözlük döndürür."""
        if self.df is None:
            return {}
        df = self.df
        gen_normal = float(((df["abi_right"] >= 0.9) & (df["abi_right"] <= 1.4)).mean())
        cc_normal  = float(((df["abi_right"] >= 1.0) & (df["abi_right"] <= 1.3)).mean())

        m_abi = df[df["sex"] == "E"]["abi_right"].dropna()
        f_abi = df[df["sex"] == "K"]["abi_right"].dropna()

        age_groups = (
            df.groupby("age_group", observed=True)["abi_right"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "ort", "std": "sd", "count": "n"})
            .to_dict(orient="index")
        )

        # Boy regresyonu
        X = df["height_cm"].values.reshape(-1, 1)
        y = df["abi_right"].values
        mask = ~np.isnan(y) & ~np.isnan(X.ravel())
        slope, r2 = np.nan, np.nan
        if mask.sum() > 10:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(X[mask], y[mask])
            slope = float(reg.coef_[0])
            r2    = float(reg.score(X[mask], y[mask]))

        corr_bmi = float(stats.pearsonr(df["bmi"].dropna(),
                                        df["abi_right"][df["bmi"].notna()])[0])
        corr_iad = float(stats.pearsonr(df["iad"].dropna(),
                                        df["abi_right"][df["iad"].notna()])[0])

        gender_p = float(stats.ttest_ind(m_abi, f_abi, equal_var=False).pvalue) if (len(m_abi) > 1 and len(f_abi) > 1) else np.nan

        self._summary_cache = {
            "label":          self.study_label,
            "n":              int(len(df)),
            "age_mean":       float(df["age"].mean()),
            "age_std":        float(df["age"].std()),
            "bmi_mean":       float(df["bmi"].mean()),
            "bmi_std":        float(df["bmi"].std()),
            "abi_right_mean": float(df["abi_right"].mean()),
            "abi_right_std":  float(df["abi_right"].std()),
            "abi_left_mean":  float(df["abi_left"].mean()),
            "abi_left_std":   float(df["abi_left"].std()),
            "iad_mean":       float(df["iad"].mean()),
            "iad_std":        float(df["iad"].std()),
            "gen_normal_pct": gen_normal * 100,
            "cc_normal_pct":  cc_normal  * 100,
            "abi_class_counts":    df["abi_class"].value_counts().to_dict(),
            "abi_class_cc_counts": df["abi_class_cc"].value_counts().to_dict(),
            "iad_class_counts":    df["iad_class"].value_counts().to_dict(),
            "low_abi_n":      int(df["low_abi_flag"].sum()),
            "male_abi_mean":  float(m_abi.mean()) if len(m_abi) else np.nan,
            "male_abi_std":   float(m_abi.std())  if len(m_abi) else np.nan,
            "male_n":         int(len(m_abi)),
            "female_abi_mean":float(f_abi.mean()) if len(f_abi) else np.nan,
            "female_abi_std": float(f_abi.std())  if len(f_abi) else np.nan,
            "female_n":       int(len(f_abi)),
            "gender_diff":    float(m_abi.mean() - f_abi.mean()) if (len(m_abi) and len(f_abi)) else np.nan,
            "gender_p":       gender_p,
            "height_slope":   slope,
            "height_r2":      r2,
            "corr_bmi_abi":   corr_bmi,
            "corr_iad_abi":   corr_iad,
            "age_groups":     age_groups,
            "device_counts":  df["measurement_device"].value_counts().to_dict() if "measurement_device" in df.columns else {},
        }
        return self._summary_cache

    # ── Plotly grafikleri ───────────────────────────────────────

    def fig_abi_histogram(self) -> go.Figure:
        df = self.df
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["abi_right"].dropna(), nbinsx=40,
            marker_color="#4C72B0", opacity=0.75,
            name="ABI (sağ)"
        ))
        for x, col, dash, label in [
            (0.9, "crimson",   "dash",  "0.9 Genel alt"),
            (1.4, "crimson",   "dash",  "1.4 Genel üst"),
            (1.0, "darkgreen", "dot",   "1.0 CC alt"),
            (1.3, "darkgreen", "dot",   "1.3 CC üst"),
        ]:
            fig.add_vline(x=x, line_color=col, line_dash=dash,
                          annotation_text=label, annotation_position="top right",
                          annotation_font_size=10)
        fig.update_layout(
            title=f"ABI Dağılımı — {self.study_label}",
            xaxis_title="ABI (sağ)", yaxis_title="Frekans",
            height=320, margin=dict(t=40, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def fig_iad_histogram(self) -> go.Figure:
        df = self.df
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["iad"].dropna(), nbinsx=40,
            marker_color="#55A868", opacity=0.75,
            name="IAD (mmHg)"
        ))
        fig.add_vline(x=10, line_color="darkorange", line_dash="dash",
                      annotation_text="10 mmHg uyarı", annotation_position="top right",
                      annotation_font_size=10)
        fig.add_vline(x=20, line_color="firebrick", line_dash="dash",
                      annotation_text="20 mmHg hata", annotation_position="top right",
                      annotation_font_size=10)
        fig.update_layout(
            title=f"IAD Dağılımı — {self.study_label}",
            xaxis_title="IAD (mmHg)", yaxis_title="Frekans",
            height=320, margin=dict(t=40, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def fig_age_box(self) -> go.Figure:
        df = self.df.copy()
        df["age_group"] = df["age_group"].astype(str)
        fig = px.box(df, x="age_group", y="abi_right",
                     color="age_group", points="outliers",
                     title=f"Yaş Grubu – ABI ({self.study_label})",
                     labels={"age_group": "Yaş Grubu", "abi_right": "ABI (sağ)"},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(
            height=320, showlegend=False,
            margin=dict(t=40, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def fig_height_scatter(self) -> go.Figure:
        df = self.df
        fig = px.scatter(df, x="height_cm", y="abi_right", color="sex",
                         opacity=0.55, trendline="ols",
                         title=f"Boy–ABI İlişkisi ({self.study_label})",
                         labels={"height_cm": "Boy (cm)", "abi_right": "ABI (sağ)", "sex": "Cinsiyet"},
                         color_discrete_map={"E": "#378ADD", "K": "#ec4899"})
        fig.update_layout(
            height=340, margin=dict(t=40, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def fig_class_bar(self) -> go.Figure:
        s = self.get_summary()
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["ABI Sınıfı (Genel)", "IAD Sınıfı"])
        abi_d = s["abi_class_counts"]
        iad_d = s["iad_class_counts"]
        fig.add_bar(x=list(abi_d.keys()), y=list(abi_d.values()),
                    marker_color="#4C72B0", name="ABI", row=1, col=1)
        fig.add_bar(x=list(iad_d.keys()), y=list(iad_d.values()),
                    marker_color="#55A868", name="IAD", row=1, col=2)
        fig.update_layout(
            title_text=f"Sınıflandırma Dağılımı — {self.study_label}",
            height=320, showlegend=False,
            margin=dict(t=50, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def fig_bmi_scatter(self) -> go.Figure:
        df = self.df
        fig = px.scatter(df, x="bmi", y="abi_right", color="sex",
                         opacity=0.5, trendline="ols",
                         title=f"VKİ–ABI İlişkisi ({self.study_label})",
                         labels={"bmi": "VKİ (kg/m²)", "abi_right": "ABI (sağ)", "sex": "Cinsiyet"},
                         color_discrete_map={"E": "#378ADD", "K": "#ec4899"})
        fig.update_layout(
            height=320, margin=dict(t=40, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # ── Markdown rapor ─────────────────────────────────────────
    def build_markdown_report(self) -> str:
        s = self.get_summary()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"# ABI/IAD Analiz Raporu — {s['label']}",
            f"**Tarih:** {ts}  |  **N:** {s['n']}",
            "",
            "## Demografik Özet",
            f"| Parametre | Değer |",
            f"|-----------|-------|",
            f"| Yaş (yıl) | {s['age_mean']:.1f} ± {s['age_std']:.1f} |",
            f"| VKİ (kg/m²) | {s['bmi_mean']:.1f} ± {s['bmi_std']:.1f} |",
            f"| Erkek / Kadın | {s['male_n']} / {s['female_n']} |",
            "",
            "## ABI Ölçüm Özeti",
            f"| Taraf | Ortalama ± SD |",
            f"|-------|---------------|",
            f"| Sağ   | {s['abi_right_mean']:.3f} ± {s['abi_right_std']:.3f} |",
            f"| Sol   | {s['abi_left_mean']:.3f} ± {s['abi_left_std']:.3f} |",
            f"| IAD (mmHg) | {s['iad_mean']:.1f} ± {s['iad_std']:.1f} |",
            "",
            "## Normal Aralık Kapsamı",
            f"- **Genel (0.9–1.4):** %{s['gen_normal_pct']:.1f}",
            f"- **Cleveland Clinic (1.0–1.3):** %{s['cc_normal_pct']:.1f}",
            "",
            "## ABI Sınıflandırması (Genel)",
        ]
        for k, v in s["abi_class_counts"].items():
            lines.append(f"- {k}: {v}")
        lines += ["", "## ABI Sınıflandırması (Cleveland Clinic)"]
        for k, v in s["abi_class_cc_counts"].items():
            lines.append(f"- {k}: {v}")
        lines += ["", "## IAD Sınıflandırması"]
        for k, v in s["iad_class_counts"].items():
            lines.append(f"- {k}: {v}")
        lines += [
            "",
            "## Demografik Analizler",
            f"**Cinsiyet farkı (E–K):** {s['gender_diff']:.3f}  (p = {s['gender_p']:.3f})",
            f"**Boy–ABI eğimi:** β = {s['height_slope']:.6f}  R² = {s['height_r2']:.3f}",
            f"**VKİ–ABI korelasyonu:** r = {s['corr_bmi_abi']:.3f}",
            f"**IAD–ABI korelasyonu:** r = {s['corr_iad_abi']:.3f}",
            "",
            "## Yaş Grubu ABI Özeti",
            "| Grup | Ort. | SD | N |",
            "|------|------|----|---|",
        ]
        for grp, vals in s["age_groups"].items():
            lines.append(f"| {grp} | {vals['ort']:.3f} | {vals['sd']:.3f} | {int(vals['n'])} |")
        lines += [
            "",
            f"> ⚕️ Bu rapor yalnızca bilgilendirme amaçlıdır. "
            f"Klinik kararlar için hekim değerlendirmesi gereklidir.",
        ]
        return "\n".join(lines)

    def save_markdown_report(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = self.output_dir / f"abi_report_{self.study_label.replace(' ','_')}_{ts}.md"
        p.write_text(self.build_markdown_report(), encoding="utf-8")
        return p


# ─────────────────────────────────────────────────────────────
# KARŞILAŞTIRMA MOTORU
# ─────────────────────────────────────────────────────────────

class ABIComparator:
    """
    İki KDSABIEngine örneğini karşılaştırır.
    Her iki çalışma bağımsız çalışmış olmalıdır.
    """

    def __init__(self, engine_a: KDSABIEngine, engine_b: KDSABIEngine):
        self.a = engine_a
        self.b = engine_b
        self.sa = engine_a.get_summary()
        self.sb = engine_b.get_summary()

    def _delta(self, key: str, fmt: str = ".3f") -> str:
        va = self.sa.get(key, np.nan)
        vb = self.sb.get(key, np.nan)
        if pd.isna(va) or pd.isna(vb):
            return "—"
        d = vb - va
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:{fmt}}"

    def statistical_tests(self) -> dict:
        """İki çalışmanın ABI ve IAD dağılımlarını istatistiksel olarak karşılaştırır."""
        results = {}

        # ABI (sağ): Mann-Whitney U (dağılım şekli bilinmiyor)
        abi_a = self.a.df["abi_right"].dropna().values
        abi_b = self.b.df["abi_right"].dropna().values
        u_stat, u_p = stats.mannwhitneyu(abi_a, abi_b, alternative="two-sided")
        results["abi_mwu_stat"] = float(u_stat)
        results["abi_mwu_p"]    = float(u_p)

        # IAD: Mann-Whitney U
        iad_a = self.a.df["iad"].dropna().values
        iad_b = self.b.df["iad"].dropna().values
        u2, p2 = stats.mannwhitneyu(iad_a, iad_b, alternative="two-sided")
        results["iad_mwu_stat"] = float(u2)
        results["iad_mwu_p"]    = float(p2)

        # Normal aralık kapsamı — chi-kare (normal vs diğer)
        def chi_normal(df, lo, hi, col="abi_right"):
            norm  = int(((df[col] >= lo) & (df[col] <= hi)).sum())
            other = int(len(df) - norm)
            return [norm, other]

        cont_gen = [chi_normal(self.a.df, 0.9, 1.4),
                    chi_normal(self.b.df, 0.9, 1.4)]
        cont_cc  = [chi_normal(self.a.df, 1.0, 1.3),
                    chi_normal(self.b.df, 1.0, 1.3)]

        chi2_gen, p_gen, _, _ = stats.chi2_contingency(cont_gen)
        chi2_cc,  p_cc,  _, _ = stats.chi2_contingency(cont_cc)
        results["chi2_gen_p"] = float(p_gen)
        results["chi2_cc_p"]  = float(p_cc)
        results["chi2_gen"]   = float(chi2_gen)
        results["chi2_cc"]    = float(chi2_cc)

        # Cohen's d (ABI)
        pool_std = np.sqrt((abi_a.std()**2 + abi_b.std()**2) / 2)
        results["cohens_d_abi"] = float((abi_a.mean() - abi_b.mean()) / pool_std) if pool_std > 0 else np.nan

        return results

    def comparison_table(self) -> pd.DataFrame:
        """Yan yana karşılaştırma tablosu."""
        sa, sb = self.sa, self.sb
        tests = self.statistical_tests()

        rows = [
            ("N",                   sa["n"],                    sb["n"],                    "—"),
            ("Yaş (ort. ± SD)",     f"{sa['age_mean']:.1f} ± {sa['age_std']:.1f}",
                                    f"{sb['age_mean']:.1f} ± {sb['age_std']:.1f}",         "—"),
            ("VKİ (ort. ± SD)",     f"{sa['bmi_mean']:.1f} ± {sa['bmi_std']:.1f}",
                                    f"{sb['bmi_mean']:.1f} ± {sb['bmi_std']:.1f}",         "—"),
            ("ABI sağ (ort.)",      f"{sa['abi_right_mean']:.3f}",  f"{sb['abi_right_mean']:.3f}",
                                    f"p={tests['abi_mwu_p']:.4f}"),
            ("ABI sol (ort.)",      f"{sa['abi_left_mean']:.3f}",   f"{sb['abi_left_mean']:.3f}",  "—"),
            ("IAD (ort.)",          f"{sa['iad_mean']:.1f}",         f"{sb['iad_mean']:.1f}",
                                    f"p={tests['iad_mwu_p']:.4f}"),
            ("Normal % (Genel)",    f"%{sa['gen_normal_pct']:.1f}", f"%{sb['gen_normal_pct']:.1f}",
                                    f"p={tests['chi2_gen_p']:.4f}"),
            ("Normal % (CC)",       f"%{sa['cc_normal_pct']:.1f}",  f"%{sb['cc_normal_pct']:.1f}",
                                    f"p={tests['chi2_cc_p']:.4f}"),
            ("Cohen's d (ABI)",     "—",                            "—",
                                    f"{tests['cohens_d_abi']:.3f}"),
            ("Erkek ABI",           f"{sa['male_abi_mean']:.3f}",   f"{sb['male_abi_mean']:.3f}",  "—"),
            ("Kadın ABI",           f"{sa['female_abi_mean']:.3f}", f"{sb['female_abi_mean']:.3f}","—"),
            ("Cinsiyet farkı (E–K)",f"{sa['gender_diff']:.3f}",     f"{sb['gender_diff']:.3f}",    "—"),
            ("Boy–ABI β",           f"{sa['height_slope']:.6f}",    f"{sb['height_slope']:.6f}",   "—"),
            ("VKİ–ABI r",           f"{sa['corr_bmi_abi']:.3f}",    f"{sb['corr_bmi_abi']:.3f}",   "—"),
            ("IAD–ABI r",           f"{sa['corr_iad_abi']:.3f}",    f"{sb['corr_iad_abi']:.3f}",   "—"),
            ("Düşük ABI bayrağı",   sa["low_abi_n"],                sb["low_abi_n"],                "—"),
        ]
        return pd.DataFrame(rows, columns=["Parametre", sa["label"], sb["label"], "İstatistik"])

    # ── Karşılaştırma grafikleri ───────────────────────────────

    def fig_abi_overlay(self) -> go.Figure:
        """İki çalışmanın ABI dağılımı üst üste."""
        fig = go.Figure()
        for eng, color in [(self.a, "#4C72B0"), (self.b, "#C44E52")]:
            fig.add_trace(go.Histogram(
                x=eng.df["abi_right"].dropna(), nbinsx=40,
                name=eng.study_label, opacity=0.55,
                marker_color=color,
            ))
        for x, dash, lbl in [(0.9,"dash","0.9"),(1.4,"dash","1.4"),(1.0,"dot","1.0 CC"),(1.3,"dot","1.3 CC")]:
            fig.add_vline(x=x, line_dash=dash, line_color="gray",
                          annotation_text=lbl, annotation_font_size=9)
        fig.update_layout(
            barmode="overlay",
            title="ABI Dağılımı Karşılaştırması",
            xaxis_title="ABI (sağ)", yaxis_title="Frekans",
            height=340, margin=dict(t=45, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
        )
        return fig

    def fig_iad_overlay(self) -> go.Figure:
        fig = go.Figure()
        for eng, color in [(self.a, "#55A868"), (self.b, "#DD8452")]:
            fig.add_trace(go.Histogram(
                x=eng.df["iad"].dropna(), nbinsx=40,
                name=eng.study_label, opacity=0.55,
                marker_color=color,
            ))
        fig.add_vline(x=10, line_dash="dash", line_color="darkorange",
                      annotation_text="10 mmHg", annotation_font_size=9)
        fig.add_vline(x=20, line_dash="dash", line_color="firebrick",
                      annotation_text="20 mmHg", annotation_font_size=9)
        fig.update_layout(
            barmode="overlay",
            title="IAD Dağılımı Karşılaştırması",
            xaxis_title="IAD (mmHg)", yaxis_title="Frekans",
            height=320, margin=dict(t=45, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
        )
        return fig

    def fig_key_metrics_bar(self) -> go.Figure:
        """Temel metrik karşılaştırması (grouped bar)."""
        sa, sb = self.sa, self.sb
        metriks = ["ABI sağ", "ABI sol", "IAD/10"]
        val_a = [sa["abi_right_mean"], sa["abi_left_mean"], sa["iad_mean"] / 10]
        val_b = [sb["abi_right_mean"], sb["abi_left_mean"], sb["iad_mean"] / 10]

        fig = go.Figure()
        fig.add_bar(x=metriks, y=val_a, name=sa["label"], marker_color="#4C72B0")
        fig.add_bar(x=metriks, y=val_b, name=sb["label"], marker_color="#C44E52")
        fig.update_layout(
            barmode="group",
            title="Temel Metrik Karşılaştırması",
            yaxis_title="Değer",
            height=300, margin=dict(t=45, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.25),
        )
        return fig

    def fig_normal_range_bar(self) -> go.Figure:
        """Normal aralık kapsamı karşılaştırması."""
        sa, sb = self.sa, self.sb
        fig = go.Figure()
        for sa_, sb_, lbl in [
            ([sa["gen_normal_pct"], sb["gen_normal_pct"]], "Genel (0.9–1.4)"),
        ]:
            pass
        labels  = ["Genel (0.9–1.4)", "Cleveland Clinic (1.0–1.3)"]
        vals_a  = [sa["gen_normal_pct"], sa["cc_normal_pct"]]
        vals_b  = [sb["gen_normal_pct"], sb["cc_normal_pct"]]
        fig.add_bar(x=labels, y=vals_a, name=sa["label"], marker_color="#4C72B0")
        fig.add_bar(x=labels, y=vals_b, name=sb["label"], marker_color="#C44E52")
        fig.update_layout(
            barmode="group",
            title="Normal Aralık Kapsamı (%)",
            yaxis_title="%",
            height=280, margin=dict(t=45, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.3),
        )
        return fig

    def fig_age_group_comparison(self) -> go.Figure:
        """Her iki çalışmada yaş grubu ortalamaları."""
        sa, sb = self.sa, self.sb
        groups = list(sa["age_groups"].keys())
        fig = go.Figure()
        fig.add_bar(
            x=groups,
            y=[sa["age_groups"][g]["ort"] for g in groups],
            error_y=dict(type="data", array=[sa["age_groups"][g]["sd"] for g in groups]),
            name=sa["label"], marker_color="#4C72B0"
        )
        fig.add_bar(
            x=groups,
            y=[sb["age_groups"][g]["ort"] for g in groups],
            error_y=dict(type="data", array=[sb["age_groups"][g]["sd"] for g in groups]),
            name=sb["label"], marker_color="#C44E52"
        )
        fig.update_layout(
            barmode="group",
            title="Yaş Grubu – ABI (Ortalama ± SD)",
            xaxis_title="Yaş Grubu", yaxis_title="ABI (sağ)",
            height=320, margin=dict(t=45, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.3),
        )
        return fig

    def fig_abi_class_compare(self) -> go.Figure:
        """ABI sınıf dağılımını iki çalışma için karşılaştır."""
        sa, sb = self.sa, self.sb
        all_classes = sorted(set(list(sa["abi_class_counts"]) + list(sb["abi_class_counts"])))
        fig = go.Figure()
        fig.add_bar(
            x=all_classes,
            y=[sa["abi_class_counts"].get(c, 0) for c in all_classes],
            name=sa["label"], marker_color="#4C72B0"
        )
        fig.add_bar(
            x=all_classes,
            y=[sb["abi_class_counts"].get(c, 0) for c in all_classes],
            name=sb["label"], marker_color="#C44E52"
        )
        fig.update_layout(
            barmode="group",
            title="ABI Sınıf Dağılımı Karşılaştırması",
            yaxis_title="Hasta Sayısı",
            height=300, margin=dict(t=45, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.3),
        )
        return fig

    def fig_violin_abi(self) -> go.Figure:
        """Violin plot: ABI dağılım şekli karşılaştırması."""
        df_a = self.a.df[["abi_right"]].copy()
        df_a["Çalışma"] = self.a.study_label
        df_b = self.b.df[["abi_right"]].copy()
        df_b["Çalışma"] = self.b.study_label
        combined = pd.concat([df_a, df_b]).dropna()
        fig = px.violin(combined, x="Çalışma", y="abi_right",
                        box=True, points="outliers",
                        color="Çalışma",
                        color_discrete_map={
                            self.a.study_label: "#4C72B0",
                            self.b.study_label: "#C44E52"
                        },
                        title="ABI Dağılım Şekli (Violin)",
                        labels={"abi_right": "ABI (sağ)"})
        fig.update_layout(
            height=340, showlegend=False,
            margin=dict(t=45, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # ── Karşılaştırma Markdown raporu ─────────────────────────
    def build_comparison_report(self) -> str:
        sa, sb = self.sa, self.sb
        tests = self.statistical_tests()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        tbl = self.comparison_table()

        lines = [
            f"# ABI/IAD Karşılaştırma Raporu",
            f"**Çalışma A:** {sa['label']}  |  **Çalışma B:** {sb['label']}",
            f"**Tarih:** {ts}",
            "",
            "## Yan Yana Özet",
            tbl.to_markdown(index=False),
            "",
            "## İstatistiksel Testler",
            f"| Test | Sonuç | p değeri | Yorum |",
            f"|------|-------|----------|-------|",
            (f"| ABI Mann-Whitney U | U={tests['abi_mwu_stat']:.0f} | "
             f"p={tests['abi_mwu_p']:.4f} | "
             f"{'Anlamlı fark' if tests['abi_mwu_p'] < 0.05 else 'Anlamlı fark yok'} |"),
            (f"| IAD Mann-Whitney U | U={tests['iad_mwu_stat']:.0f} | "
             f"p={tests['iad_mwu_p']:.4f} | "
             f"{'Anlamlı fark' if tests['iad_mwu_p'] < 0.05 else 'Anlamlı fark yok'} |"),
            (f"| Chi-kare (Genel) | χ²={tests['chi2_gen']:.2f} | "
             f"p={tests['chi2_gen_p']:.4f} | "
             f"{'Anlamlı fark' if tests['chi2_gen_p'] < 0.05 else 'Anlamlı fark yok'} |"),
            (f"| Chi-kare (CC) | χ²={tests['chi2_cc']:.2f} | "
             f"p={tests['chi2_cc_p']:.4f} | "
             f"{'Anlamlı fark' if tests['chi2_cc_p'] < 0.05 else 'Anlamlı fark yok'} |"),
            f"| Cohen's d (ABI) | d={tests['cohens_d_abi']:.3f} | — | "
            + (f"Küçük etki" if abs(tests['cohens_d_abi']) < 0.2
               else "Orta etki" if abs(tests['cohens_d_abi']) < 0.5
               else "Büyük etki") + " |",
            "",
            "## Temel Bulgular",
            f"- ABI ortalamaları: **{sa['label']}** {sa['abi_right_mean']:.3f}  vs  **{sb['label']}** {sb['abi_right_mean']:.3f}",
            f"- IAD ortalamaları: **{sa['label']}** {sa['iad_mean']:.1f} mmHg  vs  **{sb['label']}** {sb['iad_mean']:.1f} mmHg",
            f"- Normal kapsam farkı (Genel): {sb['gen_normal_pct'] - sa['gen_normal_pct']:+.1f} puan",
            f"- Normal kapsam farkı (CC): {sb['cc_normal_pct'] - sa['cc_normal_pct']:+.1f} puan",
            "",
            "> ⚕️ Bu karşılaştırma raporu bilgilendirme amaçlıdır.",
        ]
        return "\n".join(lines)

    def save_comparison_report(self, output_dir: str = "abi_outputs") -> Path:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = p / f"abi_comparison_{ts}.md"
        fpath.write_text(self.build_comparison_report(), encoding="utf-8")
        return fpath
