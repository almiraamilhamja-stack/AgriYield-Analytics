"""AgriYield Analytics: Crop Yield Intelligence Dashboard (Streamlit).

Run:  python -m streamlit run app.py
Every number, chart and prediction is computed from the supplied CSV at runtime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import crop_core as cc
from crop_core import BOOL, CAT, FEATURES, NUM, TARGET

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / ".cache"
DATASET_NAME = "Agriculture Crop Yield Dataset"

# ---- palette (from the design brief) ----
FOREST, GREEN, LIGHT, BG, CARD = "#174C35", "#27864B", "#E7F3E8", "#F7F9F5", "#FFFFFF"
TEXT, MUTED, AMBER, RED = "#202820", "#778078", "#E8A23A", "#D9534F"
GREEN_MID, GREEN_SOFT, GREY = "#5DA877", "#A9D3B4", "#B8C2BA"
YLABEL = "Yield (tons per hectare)"
PAGES = ["Overview", "Exploratory Data Analysis", "Predictive Analytics",
         "Scenario Simulator", "Model Validation", "Methods & Data Quality"]
DISCLAIMER = ("Scenario results are model-based estimates derived from observational data. They do not "
              "establish causation and should not be interpreted as guaranteed yield improvements.")

st.set_page_config(page_title="AgriYield Analytics", page_icon="🌿", layout="wide")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
_HILLS = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1400 200' "
          "preserveAspectRatio='xMidYMid slice'><defs><linearGradient id='s' x1='0' y1='0' x2='0' y2='1'>"
          "<stop offset='0' stop-color='rgb(208,232,244)'/><stop offset='1' stop-color='rgb(240,247,238)'/></linearGradient>"
          "<clipPath id='c'><path d='M0 150 C300 122 600 152 900 130 S1250 122 1400 140 V200 H0Z'/></clipPath></defs><rect width='1400' height='200' fill='url(%23s)'/>"
          "<path d='M0 112 L120 72 L240 106 L380 60 L520 100 L680 54 L820 98 L980 64 L1120 100 L1260 70 L1400 106 V200 H0Z' "
          "fill='rgb(150,184,186)' opacity='0.7'/>"
          "<path d='M0 126 C250 90 450 130 700 104 S1150 86 1400 116 V200 H0Z' fill='rgb(138,188,138)'/>"
          "<path d='M0 150 C300 122 600 152 900 130 S1250 122 1400 140 V200 H0Z' fill='rgb(96,160,92)'/><g clip-path='url(%23c)'><line x1='-700' y1='200' x2='392' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='-625' y1='200' x2='408' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='-550' y1='200' x2='425' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='-475' y1='200' x2='442' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='-400' y1='200' x2='458' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='-325' y1='200' x2='474' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='-250' y1='200' x2='491' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='-175' y1='200' x2='508' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='-100' y1='200' x2='524' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='-25' y1='200' x2='540' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='50' y1='200' x2='557' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='125' y1='200' x2='574' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='200' y1='200' x2='590' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='275' y1='200' x2='606' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='350' y1='200' x2='623' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='425' y1='200' x2='640' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='500' y1='200' x2='656' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='575' y1='200' x2='672' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='650' y1='200' x2='689' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='725' y1='200' x2='706' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='800' y1='200' x2='722' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='875' y1='200' x2='738' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='950' y1='200' x2='755' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1025' y1='200' x2='772' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1100' y1='200' x2='788' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1175' y1='200' x2='804' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1250' y1='200' x2='821' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1325' y1='200' x2='838' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1400' y1='200' x2='854' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1475' y1='200' x2='870' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1550' y1='200' x2='887' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1625' y1='200' x2='904' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1700' y1='200' x2='920' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1775' y1='200' x2='936' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1850' y1='200' x2='953' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='1925' y1='200' x2='970' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='2000' y1='200' x2='986' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='2075' y1='200' x2='1002' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/><line x1='2150' y1='200' x2='1019' y2='128' stroke='rgb(52,118,66)' stroke-width='2.2' opacity='0.55'/></g></svg>")

CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"] { background: __BG__; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1560px; }
html, body, [class*="css"] { color: __TEXT__; font-family: "Segoe UI", system-ui, -apple-system, sans-serif; }
h1, h2, h3, h4 { color: __TEXT__; letter-spacing: -0.01em; }

/* Sidebar */
[data-testid="stSidebar"] { background: __FOREST__; min-width: 280px; }
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] summary p { color: #E8F2EA; }
[data-testid="stSidebar"] input { color: __TEXT__ !important; background: #fff !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: __TEXT__ !important; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] div[role="radiogroup"] label { width: 100%; padding: 11px 14px; border-radius: 10px; cursor: pointer; }
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] label p { color: #fff; font-size: 0.97rem; font-weight: 500; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,0.08); }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background: #3F9462; }
.brand { display:flex; align-items:center; gap:12px; margin: 4px 0 18px 4px; }
.brand b { color:#fff; font-size:1.45rem; line-height:1.1; font-family: Georgia, serif; } .brand i { font-style:normal; }
.status-box { border:1px solid rgba(255,255,255,0.25); border-radius:12px; padding:12px 14px; color:#E8F2EA; font-size:0.86rem; margin-top:14px; }
.status-box .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; }

/* Banner */
.banner { background: linear-gradient(90deg, rgba(247,249,245,0.97) 0%, rgba(247,249,245,0.80) 55%, rgba(247,249,245,0.35) 100%),
          url("__HILLS__") center/cover no-repeat; border:1px solid #E1E8DD; border-radius:16px;
          padding: 18px 24px; display:flex; flex-wrap:wrap; gap:18px; align-items:center; justify-content:space-between; }
.banner .left { display:flex; align-items:center; gap:16px; }
.banner h1 { margin:0; font-size:2.1rem; line-height:1.1; font-weight:700; }
.banner .sub { font-size:1.2rem; font-weight:600; color:__FOREST__; }
.banner .desc { font-size:0.85rem; color:__MUTED__; margin-top:2px; }
.banner .right { display:flex; flex-wrap:wrap; gap:12px; }
.mini { background:#fff; border:1px solid #E1E8DD; border-radius:12px; padding:10px 16px; font-size:0.82rem; box-shadow:0 1px 3px rgba(23,76,53,0.08); }
.mini b { display:block; font-size:0.86rem; }
.pill { display:inline-block; background:__LIGHT__; color:__GREEN__; border-radius:999px; padding:2px 10px; font-weight:600; font-size:0.78rem; }
.pill.warn { background:#FFF3E0; color:#B26A00; } .pill.bad { background:#FDECEA; color:__RED__; }

/* Cards */
.kpi { background:#fff; border:1px solid #E6EBE3; border-radius:14px; padding:16px 18px; display:flex; gap:14px;
       align-items:center; box-shadow:0 1px 4px rgba(23,76,53,0.07); min-height:112px; }
.kpi-ic { background:__LIGHT__; color:__GREEN__; border-radius:12px; width:52px; height:52px; display:flex; align-items:center; justify-content:center; flex:none; }
.kpi-t { font-size:0.84rem; color:__TEXT__; font-weight:600; }
.kpi-v { font-size:1.9rem; font-weight:700; line-height:1.15; }
.kpi-s { font-size:0.8rem; color:__MUTED__; }
.find { background:#fff; border:1px solid #E6EBE3; border-radius:14px; padding:16px 18px; box-shadow:0 1px 4px rgba(23,76,53,0.07); height:100%; }
.find h4 { margin:0 0 4px 0; font-size:1rem; display:flex; align-items:center; gap:8px; }
.find .big { font-size:1.5rem; font-weight:700; color:__GREEN__; }
.find p { margin:4px 0; font-size:0.9rem; } .find .why { color:__MUTED__; font-size:0.82rem; }
.explain { background:#FBFCF9; border:1px solid #E1E8DD; border-radius:10px; padding:9px 14px; font-size:0.88rem; }
.section-h { font-size:1.15rem; font-weight:700; margin: 18px 0 8px 0; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:14px; box-shadow:0 1px 4px rgba(23,76,53,0.07); }
div[data-testid="stMetric"] { background:#fff; border:1px solid #E6EBE3; border-radius:12px; padding:10px 14px; }
@media (max-width: 900px) { .banner h1 { font-size:1.6rem; } .kpi-v { font-size:1.5rem; } }

/* ---- v3 overrides ---- */
h1, h2, h3, .banner h1, .banner .sub, .section-h, .ch-t, .find h4, .kpi-t { font-family: Georgia, "Times New Roman", serif; }
.banner { background: linear-gradient(90deg, rgba(247,249,245,0.82) 0%, rgba(247,249,245,0.40) 45%, rgba(247,249,245,0) 100%),
          url("__HILLS__") center/cover no-repeat; min-height: 150px; padding: 22px 28px; border-radius: 16px; }
.banner h1 { font-size: 2.5rem; } .banner .sub { font-size: 1.35rem; }
.brand b span { color:#8FD3A4; } .brand .tag { color:#BFD8C8; font-size:0.78rem; font-weight:400; display:block; }
[data-testid="stSidebar"] .stButton button { background: transparent; border: 1px solid rgba(255,255,255,0.45); color:#fff; width:100%; border-radius:10px; }
[data-testid="stSidebar"] .stButton button p { color:#fff; }
[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,0.12); border-color:#fff; }
.info-box { border:1px solid rgba(255,255,255,0.25); border-radius:14px; padding:14px 16px; color:#E8F2EA; font-size:0.86rem; margin-top:6px; }
.info-box h5 { color:#fff; font-size:0.95rem; margin:0 0 8px 0; font-family: Georgia, serif; }
.info-box hr { border:0; border-top:1px solid rgba(255,255,255,0.2); margin:10px 0; }
.kpi-tag { margin-left:auto; align-self:flex-start; font-size:0.74rem; font-weight:700; color:__GREEN__; background:__LIGHT__; border-radius:999px; padding:2px 9px; white-space:nowrap; }
.ch-t { font-size:1.08rem; font-weight:700; line-height:1.2; } .ch-s { font-size:0.8rem; color:__MUTED__; }
.chip { display:inline-block; background:#F1F6EF; border:1px solid #DDE7D9; border-radius:999px; padding:3px 12px; margin:3px 4px 3px 0; font-size:0.82rem; }
.tgt { background:__LIGHT__; border-radius:12px; padding:12px 14px; display:flex; gap:12px; align-items:center; }
.tgt b { display:block; } .tgt span { font-size:0.8rem; color:__MUTED__; }
.panel-p { font-size:0.88rem; color:#4A544B; margin:6px 0 0 0; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div { min-height: 36px; font-size: 0.85rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1)::before { content:''; width:22px; height:22px; margin-right:12px; flex:none; background:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M3 11l9-8 9 8v10H3z M9 21v-6h6v6'/></svg>") center/contain no-repeat; }
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2)::before { content:''; width:22px; height:22px; margin-right:12px; flex:none; background:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M5 20V10 M12 20V4 M19 20v-7'/></svg>") center/contain no-repeat; }
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3)::before { content:''; width:22px; height:22px; margin-right:12px; flex:none; background:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M3 17l6-6 4 4 8-8 M15 7h6v6'/></svg>") center/contain no-repeat; }
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4)::before { content:''; width:22px; height:22px; margin-right:12px; flex:none; background:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M4 7h10 M18 7h2 M4 17h2 M10 17h10 M14 5v4 M8 15v4'/></svg>") center/contain no-repeat; }
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(5)::before { content:''; width:22px; height:22px; margin-right:12px; flex:none; background:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z M9 12l2 2 4-4'/></svg>") center/contain no-repeat; }
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(6)::before { content:''; width:22px; height:22px; margin-right:12px; flex:none; background:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M7 3h7l4 4v14H7z M14 3v4h4 M10 12h5 M10 16h5'/></svg>") center/contain no-repeat; }
</style>
"""
for _k, _v in {"__BG__": BG, "__TEXT__": TEXT, "__FOREST__": FOREST, "__MUTED__": MUTED, "__LIGHT__": LIGHT,
               "__GREEN__": GREEN, "__RED__": RED, "__HILLS__": _HILLS}.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)


def _svg(inner: str, size: int = 26) -> str:
    return (f"<svg viewBox='0 0 24 24' width='{size}' height='{size}' fill='none' stroke='currentColor' "
            f"stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'>{inner}</svg>")


ICON = {
    "doc": _svg("<path d='M7 3h7l4 4v14H7z M14 3v4h4 M10 12h5 M10 16h5'/>"),
    "sprout": _svg("<path d='M12 21v-8 M12 13c0-4-3-6-7-6 0 4 3 6 7 6z M12 15c0-3 2-5 6-5 0 3-2 5-6 5z'/>"),
    "award": _svg("<path d='M8 4h8v5a4 4 0 0 1-8 0z M8 6H5a3 3 0 0 0 3 4 M16 6h3a3 3 0 0 1-3 4 M12 13v4 M9 20h6'/>"),
    "chart": _svg("<path d='M5 20V10 M12 20V4 M19 20v-7'/>"),
    "rain": _svg("<path d='M7 15a4 4 0 0 1 0-8 5 5 0 0 1 9.5 1A3.5 3.5 0 0 1 17 15z M9 18l-1 2 M13 18l-1 2 M17 18l-1 2'/>", 22),
    "alert": _svg("<path d='M12 4l9 16H3z M12 10v4 M12 17v.5'/>", 22),
    "db": _svg("<ellipse cx='12' cy='6' rx='7' ry='3'/><path d='M5 6v12c0 1.7 3 3 7 3s7-1.3 7-3V6 M5 12c0 1.7 3 3 7 3s7-1.3 7-3'/>", 22),
    "cal": _svg("<path d='M5 6h14v14H5z M5 10h14 M9 3v4 M15 3v4'/>", 22),
}
LEAF = _svg("<path d='M5 19C5 10 10 5 20 4c0 10-5 15-13 15 M5 19c3-5 6-8 10-10'/>", 44)


def kpi_card(title: str, value: str, sub: str, icon: str, tag: str = "") -> str:
    t = f"<div class='kpi-tag'>{tag}</div>" if tag else ""
    return (f"<div class='kpi'><div class='kpi-ic'>{ICON[icon]}</div><div><div class='kpi-t'>{title}</div>"
            f"<div class='kpi-v'>{value}</div><div class='kpi-s'>{sub}</div></div>{t}</div>")


def finding_card(icon: str, title: str, big: str, text: str, why: str) -> str:
    return (f"<div class='find'><h4><span style='color:{GREEN}'>{ICON[icon]}</span>{title}</h4>"
            f"<div class='big'>{big}</div><p>{text}</p><p class='why'>{why}</p></div>")


def explain(shows: str, matters: str, nxt: str = "") -> None:
    extra = f"<br><b>Next step:</b> {nxt}" if nxt else ""
    st.markdown(f"<div class='explain'><b>What it shows:</b> {shows}<br><b>Why it matters:</b> {matters}{extra}</div>",
                unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=48 if title else 20, b=10),
                      paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(size=12, color=TEXT),
                      title=dict(text=title, x=0.01, font=dict(size=15)) if title else None,
                      colorway=[GREEN, GREEN_MID, GREEN_SOFT, AMBER, GREY],
                      hoverlabel=dict(bgcolor="white", font_size=12))
    fig.update_xaxes(gridcolor="#E9EEE6", zeroline=False)
    fig.update_yaxes(gridcolor="#E9EEE6", zeroline=False)
    return fig


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# --------------------------------------------------------------------------- #
# Cached loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading and validating the dataset…")
def cached_data(path: str, stamp: float):
    df = cc.load_data(path)
    return df, cc.quality_report(df)


@st.cache_resource(show_spinner="Training models once (about a minute for 1M rows)…")
def cached_bundle(path: str, stamp: float):
    df, _ = cached_data(path, stamp)
    return cc.get_or_train(df, path, CACHE_DIR)


def find_default_csv() -> str:
    for rel in ("Cleaned_Crop_Yield.csv", "data/Cleaned_Crop_Yield.csv"):
        if (APP_DIR / rel).is_file():
            return str(APP_DIR / rel)
    return str(APP_DIR / "Cleaned_Crop_Yield.csv")


# --------------------------------------------------------------------------- #
# Sidebar: brand, navigation, data source, dataset status
# --------------------------------------------------------------------------- #
st.sidebar.markdown(f"<div class='brand'><span style='color:#9AD4AC'>{LEAF}</span><b>AgriYield<br><span>Analytics</span><i class='tag'>Crop Yield Intelligence</i></b></div>",
                    unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed", key="nav")

with st.sidebar.expander("Data source", expanded=False):
    csv_path = st.text_input("CSV path", value=find_default_csv())
    upload = st.file_uploader("…or upload a CSV", type="csv")
if upload is not None:
    CACHE_DIR.mkdir(exist_ok=True)
    _t = CACHE_DIR / "uploaded_Cleaned_Crop_Yield.csv"
    if not _t.exists() or _t.stat().st_size != upload.size:
        _t.write_bytes(upload.getvalue())
    csv_path = str(_t)


def sidebar_status(state: str, rows: str = "—", cols: str = "—", colour: str = AMBER, updated: str = "—") -> None:
    st.sidebar.markdown(
        "<svg viewBox='0 0 200 110' width='100%' height='90' style='opacity:0.28;margin-top:18px'><path d='M20 105C20 55 55 25 120 20c0 55-30 85-100 85z' fill='#8FD3A4'/><path d='M110 105c0-40 25-62 75-66 0 42-22 66-75 66z' fill='#6FBF89'/></svg>"
        f"<div class='info-box'><h5>Dataset Information</h5>{DATASET_NAME}<br>"
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{colour};margin-right:7px'></span>{state}<br>"
        f"{rows} rows<br>{cols} columns<hr><b>Last updated</b><br>{updated}</div>", unsafe_allow_html=True)


p = Path(csv_path).expanduser()
if not p.is_file():
    sidebar_status("Not loaded", colour=RED)
    st.title("AgriYield Analytics")
    st.error(f"Dataset not found at `{csv_path}`. Put `Cleaned_Crop_Yield.csv` next to `app.py` "
             "(or in a `data/` folder), or upload it under **Data source** in the sidebar.")
    st.stop()
try:
    STAMP = p.stat().st_mtime
    DF, QR = cached_data(str(p), STAMP)
except cc.DataError as exc:
    sidebar_status("Load failed", colour=RED)
    st.title("AgriYield Analytics")
    st.error(f"Data problem: {exc}")
    st.stop()
try:
    BUNDLE = cached_bundle(str(p), STAMP)
except cc.DataError as exc:
    sidebar_status("Model training failed", f"{len(DF):,}", str(DF.shape[1]), RED)
    st.title("AgriYield Analytics")
    st.error(f"Model problem: {exc}")
    st.stop()

META, TM, CV = BUNDLE["meta"], BUNDLE["test_metrics"], BUNDLE["cv"]
BEST = META["best_name"]
sidebar_status("Loaded (local file)", f"{QR['n_rows']:,}", str(QR["n_cols"]), GREEN_MID, META.get("trained_at", "not recorded"))
if st.sidebar.button("↻  Reload Dataset", help="Re-reads the CSV. Models are retrained only if the file changed."):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    f"""<div class='banner'>
  <div class='left'><span style='color:{GREEN}'>{LEAF}</span>
    <div><h1>AgriYield Analytics</h1><div class='sub'>Crop Yield Intelligence Dashboard</div>
    <div class='desc'>Exploring data. Predicting yields. Supporting better agricultural decisions.</div></div></div>
  <div class='right'>
    <div class='mini'><b>{ICON['db']} Dataset: {DATASET_NAME}</b>{QR['n_rows']:,} rows · {QR['n_cols']} columns
       &nbsp;<span class='pill'>Loaded</span></div>
    <div class='mini'><b>{ICON['cal']} Last updated</b>{META.get('trained_at', 'not recorded')}</div>
  </div></div>""", unsafe_allow_html=True)
st.caption("Locally loaded CSV (not a live feed). Statistics, models and predictions are recomputed from the file.")


# --------------------------------------------------------------------------- #
# EDA computations
# --------------------------------------------------------------------------- #
def subset(df: pd.DataFrame, fkey, include_neg: bool) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    for col, vals in fkey:
        m &= df[col].isin(list(vals)).to_numpy()
    if not include_neg:
        m &= (df[TARGET] >= 0).to_numpy()
    return df[m]


def _group_table(d: pd.DataFrame, col: str) -> pd.DataFrame:
    g = d.groupby(col, observed=True)[TARGET].agg(n="size", mean="mean", sd="std", median="median").reset_index()
    if col in BOOL:
        g[col] = g[col].map({True: "Yes", False: "No"})
    g[col] = g[col].astype(str)
    g["ci"] = 1.96 * g["sd"].fillna(0) / np.sqrt(g["n"])
    return g


def _binned(d: pd.DataFrame, col: str, nb: int) -> pd.DataFrame:
    b = pd.cut(d[col].astype(np.float64), bins=nb)
    g = (pd.DataFrame({"_b": b, "y": d[TARGET].astype(np.float64)}).groupby("_b", observed=True)["y"]
         .agg(n="size", mean="mean", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75)))
    g["mid"] = [i.mid for i in g.index]
    return g.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_eda(_df: pd.DataFrame, fkey, include_neg: bool, stamp_: float) -> dict | None:
    d = subset(_df, fkey, include_neg)
    if d.empty:
        return None
    y = d[TARGET].to_numpy(np.float64)
    out: dict = {"n": len(d), "mean": float(y.mean()), "median": float(np.median(y)), "sd": float(y.std()),
                 "n_neg": int((y < 0).sum()), "hist": np.histogram(y, bins=50)}
    out["corr"] = {c: (float(np.corrcoef(d[c].to_numpy(np.float64), y)[0, 1]) if len(d) > 2 else float("nan"))
                   for c in NUM}
    x = d["Rainfall_mm"].to_numpy(np.float64)
    out["rain_fit"] = tuple(np.polyfit(x, y, 1)) if len(d) > 2 else (float("nan"), float("nan"))
    out["scatter"] = d.sample(min(4000, len(d)), random_state=cc.SEED)[["Rainfall_mm", "Temperature_Celsius", TARGET]].astype(float)
    out["groups"], out["eta"] = {}, {}
    for c in CAT + BOOL:
        if d[c].nunique() > 1:
            out["groups"][c] = _group_table(d, c)
            out["eta"][c] = cc.eta_squared(d, c)
    out["bins"] = {"Rainfall_mm": _binned(d, "Rainfall_mm", 20), "Temperature_Celsius": _binned(d, "Temperature_Celsius", 20),
                   "Days_to_Harvest": _binned(d, "Days_to_Harvest", 15)}
    return out


ALL_KEY = tuple(sorted([(c, tuple(sorted(map(str, DF[c].cat.categories)))) for c in CAT] +
                       [(b, (True, False)) for b in BOOL]))
OV = compute_eda(DF, ALL_KEY, False, STAMP)


def eta_word(e: float) -> str:
    return ("negligible" if e < 0.01 else "small" if e < 0.06 else "medium" if e < 0.14 else "large")


def corr_word(r: float) -> str:
    a = abs(r)
    return "negligible" if a < 0.1 else "weak" if a < 0.4 else "moderate" if a < 0.7 else "strong"


def band_chart(b: pd.DataFrame, xlabel: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b["mid"], y=b["q75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=b["mid"], y=b["q25"], fill="tonexty", fillcolor="rgba(39,134,75,0.16)", line=dict(width=0),
                             name="Middle 50% of records", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=b["mid"], y=b["mean"], mode="lines+markers", name="Mean yield",
                             line=dict(color=GREEN), customdata=b["n"],
                             hovertemplate=xlabel + " %{x:.1f}<br>Mean %{y:.2f} t/ha<br>n=%{customdata:,}<extra></extra>"))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=YLABEL)
    return style(fig, 360, title)


def hist_chart(e: dict, title: str = "Crop Yield Distribution") -> go.Figure:
    counts, edges = e["hist"]
    mids = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=mids, y=counts, marker_color=[AMBER if m < 0 else GREEN for m in mids],
                           width=np.diff(edges) * 0.96,
                           hovertemplate="%{x:.2f} t/ha<br>%{y:,} records<extra></extra>"))
    fig.add_vline(x=e["mean"], line_dash="dash", line_color=FOREST,
                  annotation_text=f"Mean {e['mean']:.2f} · Median {e['median']:.2f}", annotation_position="top right")
    fig.update_layout(xaxis_title=YLABEL, yaxis_title="Number of records", bargap=0.02)
    return style(fig, 330, title)


def scatter_chart(e: dict, xcol: str, xlabel: str, title: str | None, with_fit: bool) -> go.Figure:
    s = e["scatter"]
    fig = go.Figure(go.Scattergl(x=s[xcol], y=s[TARGET], mode="markers", name="Records (random sample)",
                                 marker=dict(size=4, color=GREEN, opacity=0.35)))
    if with_fit and np.isfinite(e["rain_fit"][0]):
        xs = np.array([s[xcol].min(), s[xcol].max()])
        slope, icpt = e["rain_fit"]
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines", name="Linear trend (all records)",
                                 line=dict(color=FOREST, width=3)))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=YLABEL, legend=dict(orientation="h", y=1.1))
    return style(fig, 330, title)


def group_bar(g: pd.DataFrame, col: str, title: str | None, horizontal: bool = True, height: int = 300) -> go.Figure:
    g = g.sort_values("mean")
    cols = [GREEN if i == len(g) - 1 else GREEN_SOFT for i in range(len(g))]
    fig = go.Figure(go.Bar(x=g["mean"] if horizontal else g[col], y=g[col] if horizontal else g["mean"],
                           orientation="h" if horizontal else "v", marker_color=cols,
                           error_x=dict(type="data", array=g["ci"]) if horizontal else None,
                           error_y=None if horizontal else dict(type="data", array=g["ci"]),
                           customdata=np.c_[g["n"], g["ci"]],
                           hovertemplate="%{y}<br>Mean %{x:.3f} t/ha ± %{customdata[1]:.3f} (95% CI)<br>n=%{customdata[0]:,}<extra></extra>"
                           if horizontal else
                           "%{x}<br>Mean %{y:.3f} t/ha ± %{customdata[1]:.3f} (95% CI)<br>n=%{customdata[0]:,}<extra></extra>"))
    fig.update_layout(**({"xaxis_title": "Mean yield (t/ha)"} if horizontal else {"yaxis_title": "Mean yield (t/ha)"}))
    return style(fig, height, title)


def practice_chart(e: dict, title: str | None = "Fertilizer and Irrigation Comparison", height: int = 300) -> go.Figure:
    fig = go.Figure()
    for val, colour in (("No", GREEN_SOFT), ("Yes", FOREST)):
        xs, ys, cis, ns = [], [], [], []
        for c, lab in (("Fertilizer_Used", "Fertilizer"), ("Irrigation_Used", "Irrigation")):
            g = e["groups"].get(c)
            if g is not None and val in set(g[c]):
                r = g[g[c] == val].iloc[0]
                xs.append(lab); ys.append(r["mean"]); cis.append(r["ci"]); ns.append(int(r["n"]))
        fig.add_trace(go.Bar(name="Used" if val == "Yes" else "Not used", x=xs, y=ys, marker_color=colour,
                             error_y=dict(type="data", array=cis), customdata=ns,
                             hovertemplate="%{x}: " + ("used" if val == "Yes" else "not used") +
                                           "<br>Mean %{y:.3f} t/ha (95% CI)<br>n=%{customdata:,}<extra></extra>"))
    fig.update_layout(barmode="group", yaxis_title="Mean yield (t/ha)", legend=dict(orientation="h", y=1.12))
    return style(fig, height, title)


# --------------------------------------------------------------------------- #
# Model charts (shared by Predictive Analytics and Model Validation)
# --------------------------------------------------------------------------- #
def avp_chart() -> go.Figure:
    pdf = BUNDLE["plot_df"]
    lim = [float(min(pdf["actual"].min(), pdf["pred"].min())), float(max(pdf["actual"].max(), pdf["pred"].max()))]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=pdf["actual"], y=pdf["pred"], mode="markers", name="Test records",
                               marker=dict(size=4, color=GREEN, opacity=0.35)))
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Ideal prediction (y = x)",
                             line=dict(color=RED, dash="dash")))
    fig.update_layout(xaxis_title="Actual yield (t/ha)", yaxis_title="Predicted yield (t/ha)",
                      legend=dict(orientation="h", y=1.1))
    return style(fig, 340, f"Actual vs predicted ({len(pdf):,} random held-out records)")


def resid_hist_chart() -> go.Figure:
    pdf = BUNDLE["plot_df"]
    fig = go.Figure(go.Histogram(x=pdf["residual"], nbinsx=50, marker_color=GREEN))
    fig.add_vline(x=0, line_dash="dash", line_color=RED)
    fig.update_layout(xaxis_title="Residual = actual − predicted (t/ha)", yaxis_title="Records")
    return style(fig, 320, "Residual distribution")


def resid_pred_chart() -> go.Figure:
    pdf = BUNDLE["plot_df"]
    fig = go.Figure(go.Scattergl(x=pdf["pred"], y=pdf["residual"], mode="markers",
                                 marker=dict(size=4, color=GREEN, opacity=0.35)))
    fig.add_hline(y=0, line_dash="dash", line_color=RED)
    fig.update_layout(xaxis_title="Predicted yield (t/ha)", yaxis_title="Residual (t/ha)")
    return style(fig, 320, "Residuals vs predicted values")


def importance_chart() -> go.Figure:
    imp = BUNDLE["importance"].sort_values("R2 drop")
    fig = go.Figure(go.Bar(x=imp["R2 drop"], y=imp["Feature"], orientation="h", marker_color=GREEN,
                           error_x=dict(type="data", array=imp["sd"])))
    fig.update_layout(xaxis_title="Drop in test R² when the feature is shuffled")
    return style(fig, 340, "Permutation importance")


def model_table() -> pd.DataFrame:
    cv = CV.set_index("Model")
    rows = []
    for name, m in TM.items():
        rows.append({"Model": name + ("  (selected)" if name == BEST else ""),
                     "Test MAE": round(m["MAE"], 3), "Test RMSE": round(m["RMSE"], 3), "Test R²": round(m["R2"], 4),
                     "CV RMSE (train)": round(cv.loc[name, "CV RMSE"], 3),
                     "CV RMSE sd": round(cv.loc[name, "CV RMSE sd"], 4), "CV R² (train)": round(cv.loc[name, "CV R2"], 4)})
    return pd.DataFrame(rows)


METRIC_HELP = (
    "- **MAE** – the average absolute prediction error, in tons per hectare.\n"
    "- **RMSE** – prediction error in which larger errors are penalised more heavily; never smaller than MAE.\n"
    "- **R²** – the proportion of yield variation explained, relative to a baseline that always predicts the mean "
    "(0 = no better than the mean).")


# =========================================================================== #
# PAGE: Overview
# =========================================================================== #
def fkey_with(col: str, val: str | None):
    base = dict(ALL_KEY)
    if val:
        base[col] = (val,)
    return tuple(sorted(base.items()))


@st.cache_data(show_spinner=False)
def quick_overview(_df: pd.DataFrame, fkey, stamp_: float) -> dict | None:
    d = subset(_df, fkey, False)
    if d.empty:
        return None
    y = d[TARGET].to_numpy(np.float64)
    x = d["Rainfall_mm"].to_numpy(np.float64)
    return {"n": len(d), "mean": float(y.mean()), "median": float(np.median(y)), "hist": np.histogram(y, bins=50),
            "corr": float(np.corrcoef(x, y)[0, 1]) if len(d) > 2 else float("nan"),
            "rain_fit": tuple(np.polyfit(x, y, 1)) if len(d) > 2 else (float("nan"), float("nan")),
            "scatter": d.sample(min(4000, len(d)), random_state=cc.SEED)[["Rainfall_mm", "Temperature_Celsius", TARGET]].astype(float)}


@st.cache_data(show_spinner=False)
def quick_groups(_df: pd.DataFrame, fkey, stamp_: float) -> dict:
    d = subset(_df, fkey, False)
    return {c: _group_table(d, c) for c in CAT + BOOL if len(d) and d[c].nunique() > 0}


def chart_head(title: str, sub: str = "") -> str:
    return f"<div class='ch-t'>{title}</div><div class='ch-s'>{sub}</div>"


def page_overview() -> None:
    best_g = OV["groups"]["Crop"].sort_values("mean", ascending=False)
    top, low = best_g.iloc[0], best_g.iloc[-1]
    gap = top["mean"] - low["mean"]
    base_mae, sel = TM["Baseline (training mean)"]["MAE"], TM[BEST]
    c = st.columns(4)
    c[0].markdown(kpi_card("Total Valid Records", f"{QR['n_valid']:,}",
                           f"of {QR['n_rows']:,} rows · valid = complete, numeric, yield ≥ 0", "doc",
                           f"{QR['n_valid'] / QR['n_rows']:.1%} valid"), unsafe_allow_html=True)
    c[1].markdown(kpi_card("Average Crop Yield", f"{OV['mean']:.2f}", f"tons per hectare · median {OV['median']:.2f}", "sprout",
                           f"SD {OV['sd']:.2f}"), unsafe_allow_html=True)
    c[2].markdown(kpi_card("Best Performing Crop", str(top["Crop"]), f"{top['mean']:.2f} tons per hectare (highest mean)", "award",
                           f"gap {gap:.2f} t/ha"), unsafe_allow_html=True)
    c[3].markdown(kpi_card("Model Predictive Performance", f"{sel['R2']:.2f}", f"R² · {BEST} · held-out test", "chart",
                           f"MAE −{1 - sel['MAE'] / base_mae:.0%} vs baseline"), unsafe_allow_html=True)

    st.markdown("<div class='section-h'>Key Findings</div>", unsafe_allow_html=True)
    r = OV["corr"]["Rainfall_mm"]
    eta = {k: v for k, v in OV["eta"].items() if k in CAT}
    kmax = max(eta, key=eta.get)
    f = st.columns(3)
    f[0].markdown(finding_card(
        "rain", "Rainfall and Crop Yield", f"Correlation: {r:.2f}",
        f"A {corr_word(r)} {'positive' if r > 0 else 'negative'} linear association between rainfall and yield.",
        f"Rainfall alone accounts for about {r * r:.0%} of yield variation (linear fit). Association, not proof of cause."),
        unsafe_allow_html=True)
    gtxt = ("Mean yield is practically the same across crops, soils, regions and weather conditions."
            if eta[kmax] < 0.01 else f"{kmax.replace('_', ' ')} shows a {eta_word(eta[kmax])} difference in mean yield.")
    f[1].markdown(finding_card(
        "sprout", "Differences Across Agricultural Groups", f"η² max = {eta[kmax]:.2%}", gtxt,
        f"η² is the share of yield variance explained by group ({kmax.replace('_', ' ')} is the largest; "
        "below 1% is negligible)."), unsafe_allow_html=True)
    f[2].markdown(finding_card(
        "alert", "Data Quality and Unusual Yield Values", f"{QR['pct_negative']:.3f}% flagged",
        f"{QR['n_negative']:,} records have negative yield"
        + (f", all at rainfall ≤ {QR['neg_rain_max']:.0f} mm." if QR["n_negative"] else "."),
        "Negative yields are physically impossible: excluded from model fitting and primary test metrics; "
        "a sensitivity check is reported."), unsafe_allow_html=True)

    st.write("")
    crops = sorted(map(str, DF["Crop"].cat.categories))
    soils = sorted(map(str, DF["Soil_Type"].cat.categories))
    a, b = st.columns(2)
    with a.container(border=True):
        h, s_, dl = st.columns([3.2, 1.4, 0.8])
        crop_sel = s_.selectbox("Crop", ["All crops"] + crops, label_visibility="collapsed", key="ov_h_crop")
        Q = quick_overview(DF, fkey_with("Crop", None if crop_sel == "All crops" else crop_sel), STAMP)
        h.markdown(chart_head("Crop Yield Distribution", "Distribution of yield values (tons per hectare), valid records"),
                   unsafe_allow_html=True)
        if Q is not None:
            cnt, edg = Q["hist"]
            dl.download_button("⬇ CSV", pd.DataFrame({"bin_start": edg[:-1], "bin_end": edg[1:], "records": cnt}).to_csv(index=False),
                               "yield_distribution.csv", "text/csv", key="dl_hist", help="Download this chart's data")
            plot(hist_chart(Q, None))
            st.caption(f"n = {Q['n']:,} · Mean {Q['mean']:.2f} · Median {Q['median']:.2f} t/ha")
    with b.container(border=True):
        h, pill, s_, dl = st.columns([2.3, 1.3, 1.4, 0.8])
        crop_sel2 = s_.selectbox("Crop", ["All crops"] + crops, label_visibility="collapsed", key="ov_s_crop")
        Q2 = quick_overview(DF, fkey_with("Crop", None if crop_sel2 == "All crops" else crop_sel2), STAMP)
        h.markdown(chart_head("Rainfall vs. Crop Yield", "Relationship between rainfall and crop yield"), unsafe_allow_html=True)
        if Q2 is not None:
            pill.markdown(f"<span class='pill'>Correlation: {Q2['corr']:.2f}</span>", unsafe_allow_html=True)
            dl.download_button("⬇ CSV", Q2["scatter"].to_csv(index=False), "rainfall_vs_yield_sample.csv", "text/csv",
                               key="dl_sc", help="Download the plotted sample")
            plot(scatter_chart(Q2, "Rainfall_mm", "Rainfall (mm)", None, True))
            st.caption(f"Points: random sample of {len(Q2['scatter']):,}; trend line fitted to all {Q2['n']:,} valid records.")

    d1, d2, d3 = st.columns(3)
    with d1.container(border=True):
        h, s_, dl = st.columns([2.4, 1.6, 0.9])
        soil_sel = s_.selectbox("Soil", ["All soil types"] + soils, label_visibility="collapsed", key="ov_c_soil")
        G = quick_groups(DF, fkey_with("Soil_Type", None if soil_sel == "All soil types" else soil_sel), STAMP)
        h.markdown(chart_head("Average Yield by Crop Type", "Mean yield, 95% CI"), unsafe_allow_html=True)
        dl.download_button("⬇ CSV", G["Crop"].round(4).to_csv(index=False), "yield_by_crop.csv", "text/csv", key="dl_crop")
        plot(group_bar(G["Crop"], "Crop", None, height=290))
    with d2.container(border=True):
        h, s_, dl = st.columns([2.4, 1.6, 0.9])
        crop_sel3 = s_.selectbox("Crop", ["All crops"] + crops, label_visibility="collapsed", key="ov_soil_crop")
        G = quick_groups(DF, fkey_with("Crop", None if crop_sel3 == "All crops" else crop_sel3), STAMP)
        h.markdown(chart_head("Average Yield by Soil Type", "Mean yield, 95% CI"), unsafe_allow_html=True)
        dl.download_button("⬇ CSV", G["Soil_Type"].round(4).to_csv(index=False), "yield_by_soil.csv", "text/csv", key="dl_soil")
        plot(group_bar(G["Soil_Type"], "Soil_Type", None, height=290))
    with d3.container(border=True):
        h, s_, dl = st.columns([2.4, 1.6, 0.9])
        crop_sel4 = s_.selectbox("Crop", ["All crops"] + crops, label_visibility="collapsed", key="ov_p_crop")
        G = quick_groups(DF, fkey_with("Crop", None if crop_sel4 == "All crops" else crop_sel4), STAMP)
        h.markdown(chart_head("Fertilizer and Irrigation Comparison", "Mean yield, 95% CI"), unsafe_allow_html=True)
        pt = pd.concat([G[k].assign(Practice=k.replace("_", " ")).rename(columns={k: "Group"}) for k in BOOL if k in G])
        dl.download_button("⬇ CSV", pt.round(4).to_csv(index=False), "yield_by_practice.csv", "text/csv", key="dl_prac")
        plot(practice_chart({"groups": G}, None, 290))
    st.caption("Bars show mean yield with 95% confidence intervals; check the axis, because small visual gaps can be practically "
               "negligible. Dropdowns re-compute each chart for that subset. Group differences are associations, not causal effects.")

    p1, p2, p3 = st.columns([3, 3, 2])
    with p1.container(border=True):
        st.markdown(f"<div class='ch-t'><span style='color:{GREEN}'>{ICON['alert']}</span> Problem Statement</div>"
                    "<p class='panel-p'>Agricultural productivity can vary with environmental conditions, crop type, soil type, "
                    "irrigation, fertilizer use and harvest timing. This study investigates how well these recorded conditions "
                    "predict crop yield, how reliable the predictions are, and how model-estimated yield differs across "
                    "hypothetical scenarios, using observational data.</p>", unsafe_allow_html=True)
    with p2.container(border=True):
        chips = "".join(f"<span class='chip'>{f_.replace('_', ' ')}</span>" for f_ in FEATURES)
        st.markdown(f"<div class='ch-t'><span style='color:{GREEN}'>{ICON['chart']}</span> Selected Input Variables</div>"
                    f"<p class='panel-p'>{chips}</p>", unsafe_allow_html=True)
    with p3.container(border=True):
        st.markdown(f"<div class='ch-t'><span style='color:{GREEN}'>{ICON['chart']}</span> Prediction Target</div>"
                    f"<div class='tgt' style='margin-top:8px'><span style='color:{GREEN}'>{ICON['sprout']}</span>"
                    f"<div><b>{TARGET}</b><span>Crop yield in tons per hectare</span></div></div>", unsafe_allow_html=True)


# =========================================================================== #
# PAGE: Exploratory Data Analysis
# =========================================================================== #
def page_eda() -> None:
    st.markdown("### Exploratory Data Analysis")
    with st.expander("Filters (apply to every view on this page)", expanded=False):
        fc = st.columns(3)
        flt = {}
        for i, col in enumerate(CAT):
            opts = sorted(map(str, DF[col].cat.categories))
            flt[col] = fc[i % 3].multiselect(col.replace("_", " "), opts, default=opts, key=f"f_{col}")
        for i, (col, lab) in enumerate((("Fertilizer_Used", "Fertilizer used"), ("Irrigation_Used", "Irrigation used"))):
            sel = fc[(len(CAT) + i) % 3].multiselect(lab, ["Yes", "No"], default=["Yes", "No"], key=f"f_{col}")
            flt[col] = [s == "Yes" for s in sel]
        inc_neg = st.checkbox("Include negative-yield records (default: excluded as invalid)", value=False, key="f_neg")
        st.caption("Filters change every number and chart below. Narrow selections shrink the sample, which widens "
                   "confidence intervals; results with a small n are unstable.")
    fkey = tuple(sorted((k, tuple(v)) for k, v in flt.items()))
    E = compute_eda(DF, fkey, inc_neg, STAMP)
    if E is None:
        st.warning("No records match the current filters. Widen the selection.")
        return
    m = st.columns(4)
    m[0].metric("Records selected", f"{E['n']:,}")
    m[1].metric("Mean yield (t/ha)", f"{E['mean']:.2f}")
    m[2].metric("Median yield (t/ha)", f"{E['median']:.2f}")
    m[3].metric("Std. deviation (t/ha)", f"{E['sd']:.2f}")
    if E["n"] < 1000:
        st.warning("Fewer than 1,000 records selected; summaries are unstable.")

    view = st.radio("View", ["Yield distribution", "Rainfall", "Temperature", "Crop · Soil · Region · Weather",
                             "Fertilizer & irrigation", "Days to harvest", "Data-quality summary"],
                    horizontal=True, key="eda_view")
    with st.container(border=True):
        if view == "Yield distribution":
            plot(hist_chart(E, "Distribution of yield"))
            explain(f"Yield centres near {E['median']:.2f} t/ha (SD {E['sd']:.2f}). "
                    + (f"{E['n_neg']:,} negative values are shown in amber." if E["n_neg"] else "Negative values are excluded."),
                    "The target is wide and roughly symmetric, so group averages alone leave most variation unexplained.",
                    "Check which variables explain that variation.")
        elif view == "Rainfall":
            l, r_ = st.columns(2)
            with l:
                plot(scatter_chart(E, "Rainfall_mm", "Rainfall (mm)", "Rainfall vs yield", True))
            with r_:
                plot(band_chart(E["bins"]["Rainfall_mm"], "Rainfall (mm)", "Mean yield by rainfall band"))
            rc = E["corr"]["Rainfall_mm"]
            explain(f"Correlation r = {rc:.2f} ({corr_word(rc)}). Binned means show the shape of the relationship; the band is the middle 50% of records.",
                    "Rainfall is the variable most strongly associated with yield, yet individual records vary widely around the trend.",
                    "Test whether a model using all variables predicts yield accurately (Predictive Analytics).")
        elif view == "Temperature":
            l, r_ = st.columns(2)
            with l:
                plot(scatter_chart(E, "Temperature_Celsius", "Temperature (°C)", "Temperature vs yield", False))
            with r_:
                plot(band_chart(E["bins"]["Temperature_Celsius"], "Temperature (°C)", "Mean yield by temperature band"))
            tc = E["corr"]["Temperature_Celsius"]
            explain(f"Correlation r = {tc:.2f} ({corr_word(tc)}).",
                    "Shows whether temperature carries usable signal on its own.",
                    "Let the model quantify its contribution alongside the other variables.")
        elif view == "Crop · Soil · Region · Weather":
            var = st.selectbox("Group by", CAT, format_func=lambda s: s.replace("_", " "), key="eda_group")
            g = E["groups"].get(var)
            if g is None:
                st.info("Only one group is selected, so no comparison is possible.")
            else:
                l, r_ = st.columns([3, 2])
                with l:
                    plot(group_bar(g, var, f"Mean yield by {var.replace('_', ' ').lower()} (95% CI)", height=340))
                with r_:
                    t = g[[var, "n", "mean", "median", "ci"]].sort_values("mean", ascending=False).round(3)
                    t.columns = [var.replace("_", " "), "n", "Mean", "Median", "±95% CI"]
                    st.dataframe(t, hide_index=True, use_container_width=True)
                e = E["eta"][var]
                explain(f"η² = {e:.2%} ({eta_word(e)}): the share of yield variance explained by {var.replace('_', ' ').lower()}.",
                        "Intervals show where the true group mean plausibly lies; overlapping intervals mean the data cannot separate those groups.",
                        "")
        elif view == "Fertilizer & irrigation":
            l, r_ = st.columns([3, 2])
            with l:
                plot(practice_chart(E, "Mean yield by management practice (95% CI)", 340))
            with r_:
                rows = []
                for c_ in BOOL:
                    g = E["groups"].get(c_)
                    if g is not None:
                        for _, row in g.iterrows():
                            rows.append({"Practice": c_.replace("_", " "), "Group": row[c_], "n": int(row["n"]),
                                         "Mean": round(row["mean"], 3), "±95% CI": round(row["ci"], 3)})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            etas = ", ".join(f"{c_.replace('_', ' ')} {E['eta'][c_]:.2%}" for c_ in BOOL if c_ in E["eta"])
            explain(f"Mean yield with and without each practice. Variance explained (η²): {etas}.",
                    "These are practices a manager can change, but this comparison is observational: other conditions may differ between groups.",
                    "Scenario Simulator compares predictions while holding other inputs fixed. This is still not causal evidence.")
        elif view == "Days to harvest":
            plot(band_chart(E["bins"]["Days_to_Harvest"], "Days to harvest", "Mean yield by days to harvest"))
            dc = E["corr"]["Days_to_Harvest"]
            explain(f"Correlation r = {dc:.3f} ({corr_word(dc)}).", "Tests whether growing-period length relates to yield.", "")
        else:
            st.markdown("**Data-quality summary (full dataset, unfiltered)**")
            q = st.columns(4)
            q[0].metric("Missing cells in file", f"{QR['raw_missing']:,}")
            q[1].metric("Rows removed as unusable", f"{QR['n_dropped_unusable']:,}")
            q[2].metric("Duplicate rows", f"{QR['n_duplicates']:,}")
            q[3].metric("Negative yields", f"{QR['n_negative']:,}")
            if QR["n_negative"]:
                nb = QR["neg_by_rain_band"]
                fig = go.Figure(go.Bar(x=nb["Rainfall band (mm)"], y=nb["negative_rate_pct"], marker_color=AMBER,
                                       customdata=nb[["negatives", "n"]],
                                       hovertemplate="%{x} mm<br>%{y:.3f}% negative<br>%{customdata[0]} of %{customdata[1]:,}<extra></extra>"))
                fig.update_layout(xaxis_title="Rainfall band (mm)", yaxis_title="% of records negative")
                plot(style(fig, 300, "Share of negative-yield records by rainfall band"))
                explain(f"All {QR['n_negative']:,} negative records occur at rainfall ≤ {QR['neg_rain_max']:.0f} mm.",
                        "They are concentrated, not random, so predictions at low rainfall are the least reliable.",
                        "Details and handling are on the Methods & Data Quality page.")
    st.caption("Correlations and group differences describe associations in this dataset; they do not prove causation.")


# =========================================================================== #
# PAGE: Predictive Analytics
# =========================================================================== #
def page_predictive() -> None:
    st.markdown("### Predictive Analytics")
    st.markdown(
        f"**Target:** `{TARGET}` · **Inputs:** the other nine columns · **Split:** {int((1 - cc.TEST_SIZE) * 100)}/"
        f"{int(cc.TEST_SIZE * 100)} train/test, seed {cc.SEED} · **Train:** {META['n_train']:,} rows · "
        f"**Test:** {META['n_test']:,} rows. Preprocessing is fitted on training data only (inside each pipeline).")
    with st.container(border=True):
        st.markdown("**Model comparison**")
        st.dataframe(model_table(), hide_index=True, use_container_width=True)
        st.caption(f"**Test** columns are computed once on the held-out test set. **CV** columns are 5-fold cross-validation on a "
                   f"{META['cv_sample']:,}-row subsample of the training data. The preferred model (**{BEST}**) was selected by "
                   "lowest CV RMSE, not by test performance.")
        base, sel = TM["Baseline (training mean)"], TM[BEST]
        lin, gbt = TM["Linear regression"], TM["HistGradientBoosting"]
        close = abs(lin["R2"] - gbt["R2"]) < 0.005
        st.info(f"Selected model: MAE {sel['MAE']:.2f} vs baseline {base['MAE']:.2f} t/ha; R² {sel['R2']:.3f}. "
                + (f"Linear regression and gradient boosting differ by only {abs(lin['R2'] - gbt['R2']):.4f} R², so the extra complexity is not clearly justified."
                   if close else f"Linear and gradient-boosting R² differ by {abs(lin['R2'] - gbt['R2']):.4f}."))
        with st.expander("What the metrics mean"):
            st.markdown(METRIC_HELP)
    a, b = st.columns(2)
    with a.container(border=True):
        plot(avp_chart())
    with b.container(border=True):
        plot(importance_chart())
        st.caption(f"Computed on a random sample of {cc.IMPORTANCE_SAMPLE:,} test records (3 repeats). Importance shows reliance, "
                   "not causation; correlated inputs can share importance.")
    c, d = st.columns(2)
    with c.container(border=True):
        plot(resid_hist_chart())
    with d.container(border=True):
        plot(resid_pred_chart())
    st.caption("Residuals centred on zero without a visible pattern suggest no systematic bias; curved or fanning patterns "
               "would indicate problems (see Model Validation for group-level checks).")


# =========================================================================== #
# PAGE: Scenario Simulator
# =========================================================================== #
def page_scenarios() -> None:
    st.markdown("### Scenario Simulator")
    st.warning(DISCLAIMER)
    ref = cc.reference_scenario(META)
    bnd = META["bounds"]
    st.caption(f"Default reference: {ref['Crop']} · {ref['Soil_Type']} soil · {ref['Region']} · {ref['Weather_Condition']} · "
               f"dataset-median rainfall, temperature and days to harvest · no fertilizer, no irrigation. All four scenarios are editable. "
               f"Predictions use the same fitted pipeline ({BEST}) that was evaluated.")
    names = ["Reference", "Scenario A", "Scenario B", "Scenario C"]
    presets = [{}, {"Fertilizer_Used": True}, {"Irrigation_Used": True}, {"Fertilizer_Used": True, "Irrigation_Used": True}]
    scen: dict[str, dict] = {}
    cols = st.columns(4)
    for col, nm, pre in zip(cols, names, presets):
        with col.container(border=True):
            st.markdown(f"**{nm}**")
            k = nm[-1] if nm != "Reference" else "R"
            row = {}
            for c_ in ("Region", "Soil_Type", "Crop", "Weather_Condition"):
                opts = META["categories"][c_]
                row[c_] = st.selectbox(c_.replace("_", " "), opts, index=opts.index(ref[c_]), key=f"{k}_{c_}")
            row["Rainfall_mm"] = st.slider("Rainfall (mm)", float(bnd["Rainfall_mm"][0]), float(bnd["Rainfall_mm"][1]),
                                           float(ref["Rainfall_mm"]), 5.0, key=f"{k}_rain")
            row["Temperature_Celsius"] = st.slider("Temperature (°C)", float(bnd["Temperature_Celsius"][0]),
                                                   float(bnd["Temperature_Celsius"][1]), float(ref["Temperature_Celsius"]), 0.5,
                                                   key=f"{k}_temp")
            row["Days_to_Harvest"] = st.slider("Days to harvest", int(bnd["Days_to_Harvest"][0]), int(bnd["Days_to_Harvest"][1]),
                                               int(ref["Days_to_Harvest"]), 1, key=f"{k}_days")
            row["Fertilizer_Used"] = st.toggle("Fertilizer used", value=pre.get("Fertilizer_Used", False), key=f"{k}_fert")
            row["Irrigation_Used"] = st.toggle("Irrigation used", value=pre.get("Irrigation_Used", False), key=f"{k}_irr")
            warns, blocked = cc.validate_scenario(META, row)
            for w in warns:
                st.warning(w)
            row["_blocked"] = blocked
            scen[nm] = row

    ok = [n for n in names if not scen[n]["_blocked"]]
    if not ok:
        st.error("No scenario has valid inputs.")
        return
    try:
        preds = dict(zip(ok, cc.predict_rows(BUNDLE, [{k: v for k, v in scen[n].items() if k != "_blocked"} for n in ok])))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return
    hi = BUNDLE["plausibility"]["train_yield_max"]
    for n, v in preds.items():
        if v < 0 or v > hi:
            st.warning(f"{n}: predicted {v:.2f} t/ha lies outside the plausible range 0–{hi:.1f} t/ha seen in training data; do not rely on it.")

    ref_pred = preds.get("Reference")
    rows = []
    for n in names:
        v = preds.get(n)
        if v is None:
            rows.append({"Scenario": n, "Predicted yield (t/ha)": None, "Difference vs reference (t/ha)": None, "Difference (%)": "n/a (invalid inputs)"})
            continue
        diff = None if ref_pred is None else v - ref_pred
        pct = (f"{100 * diff / ref_pred:+.1f}%" if (ref_pred is not None and ref_pred > 0.01 and n != "Reference") else "n/a")
        rows.append({"Scenario": n, "Predicted yield (t/ha)": round(v, 3),
                     "Difference vs reference (t/ha)": None if (diff is None or n == "Reference") else round(diff, 3), "Difference (%)": pct})
    res = pd.DataFrame(rows)

    show_band = st.toggle("Show approximate error band (documented below)", value=True)
    q = META["resid_q"]
    cards = st.columns(4)
    for cc_, r_ in zip(cards, rows):
        val = "—" if r_["Predicted yield (t/ha)"] is None else f"{r_['Predicted yield (t/ha)']:.2f}"
        d_ = r_["Difference vs reference (t/ha)"]
        sub = "reference point" if r_["Scenario"] == "Reference" else (
            "no valid prediction" if d_ is None else f"{d_:+.2f} t/ha vs reference · {r_['Difference (%)']}")
        cc_.markdown(kpi_card(r_["Scenario"], val, sub, "sprout"), unsafe_allow_html=True)
    l, r_ = st.columns([3, 2])
    with l.container(border=True):
        xs = list(preds)
        fig = go.Figure(go.Bar(x=xs, y=[preds[n] for n in xs], marker_color=[GREY] + [GREEN] * (len(xs) - 1) if xs[0] == "Reference" else GREEN,
                               text=[f"{preds[n]:.2f}" for n in xs], textposition="outside",
                               error_y=(dict(type="data", symmetric=False, array=[q["q90"]] * len(xs), arrayminus=[-q["q10"]] * len(xs))
                                        if show_band else None)))
        fig.update_layout(yaxis_title=YLABEL)
        plot(style(fig, 340, "Predicted yield by scenario"))
    with r_.container(border=True):
        st.dataframe(res, hide_index=True, use_container_width=True)
        st.caption(f"Error band method: 10th–90th percentile of the model's held-out test residuals ({q['q10']:+.2f} to {q['q90']:+.2f} t/ha), "
                   "added to each prediction. It describes typical single-record error across the whole test set; it is not conditional "
                   "on the scenario and not a formal prediction interval. Percent differences are shown only when the reference "
                   "prediction is above 0.01 t/ha.")

    st.markdown("<div class='section-h'>Sensitivity analysis</div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    base_name = s1.selectbox("Base scenario", ok, index=min(1, len(ok) - 1), key="sens_base")
    var = s2.selectbox("Variable to vary", FEATURES, format_func=lambda s: s.replace("_", " "), key="sens_var")
    base = {k: v for k, v in scen[base_name].items() if k != "_blocked"}
    if var in NUM:
        lo, hi_ = bnd[var]
        grid = np.linspace(lo, hi_, 40)
        if var == "Days_to_Harvest":
            grid = np.unique(np.round(grid)).astype(int)
        xs, labels = list(grid), None
    elif var in BOOL:
        xs, labels = [False, True], ["No", "Yes"]
    else:
        xs = META["categories"][var]; labels = xs
    ys = cc.predict_rows(BUNDLE, [{**base, var: v} for v in xs])
    with st.container(border=True):
        if var in NUM:
            fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=GREEN, width=3), name="Predicted yield"))
            fig.add_trace(go.Scatter(x=[base[var]], y=cc.predict_rows(BUNDLE, [base]), mode="markers", name=f"{base_name} (current)",
                                     marker=dict(size=11, color=AMBER, line=dict(width=1, color=FOREST))))
            if var == "Rainfall_mm" and META["neg_rain_max"]:
                fig.add_vrect(x0=bnd[var][0], x1=META["neg_rain_max"], fillcolor=AMBER, opacity=0.15, line_width=0,
                              annotation_text="negative yields recorded", annotation_position="top left")
            fig.update_layout(xaxis_title=var.replace("_", " "), yaxis_title=YLABEL, legend=dict(orientation="h", y=1.1))
        else:
            fig = go.Figure(go.Bar(x=labels, y=ys, marker_color=GREEN, text=[f"{v:.2f}" for v in ys], textposition="outside"))
            fig.update_layout(xaxis_title=var.replace("_", " "), yaxis_title=YLABEL)
        plot(style(fig, 330, f"Predicted yield vs {var.replace('_', ' ').lower()} (other inputs from {base_name})"))
        spread = float(np.max(ys) - np.min(ys))
        noise = 0.1 * TM[BEST]["MAE"]
        st.caption(f"Predicted range across this variable: {spread:.3f} t/ha. "
                   + ("That is below 10% of the model's test MAE, so this input has a negligible effect on predictions."
                      if spread < noise else
                      f"Compare this with the model's typical error (test MAE {TM[BEST]['MAE']:.2f} t/ha). These are model-predicted associations, not causal effects.")
                   + " No fertilizer amount, irrigation schedule or planting plan is recommended: the data contain only yes/no flags.")


# =========================================================================== #
# PAGE: Model Validation
# =========================================================================== #
def page_validation() -> None:
    st.markdown("### Model Validation & Reliability")
    pl, sens = BUNDLE["plausibility"], BUNDLE["sensitivity_with_negatives"]
    st.markdown("**Validation checks that have actually run**")
    checks = [("Held-out test evaluation", True, f"{META['n_test']:,} unseen records"),
              ("5-fold cross-validation", True, f"training subsample of {META['cv_sample']:,} rows"),
              ("Residual analysis", True, "histogram and residual-vs-predicted"),
              ("Group performance", True, "crop, soil, region, weather, practices, rainfall bands"),
              ("Implausible-prediction check", True, "negative and above-training-maximum predictions"),
              ("Sensitivity including negative-yield rows", True, "test set"),
              ("External validation on independent / real farm data", False, "not performed"),
              ("Causal validation", False, "not possible with observational data")]
    st.markdown(" ".join(f"<span class='pill {'' if ok else 'warn'}'>{'✓' if ok else '○'} {n}</span>" for n, ok, _ in checks),
                unsafe_allow_html=True)
    st.caption("A trained model is not a validated model: checks marked ○ have not been done.")

    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**Baseline vs candidate models**")
        st.dataframe(model_table(), hide_index=True, use_container_width=True)
        st.caption(f"Selected by CV RMSE: **{BEST}**. Sensitivity (test set *including* negative yields): MAE {sens['MAE']:.3f}, "
                   f"RMSE {sens['RMSE']:.3f}, R² {sens['R2']:.4f}.")
    with b.container(border=True):
        st.markdown(f"**{cc.CV_FOLDS}-fold cross-validation (training data only)**")
        cvt = CV.round(4).copy()
        st.dataframe(cvt[["Model", "CV MAE", "CV RMSE", "CV RMSE sd", "CV R2", "CV R2 sd"]], hide_index=True, use_container_width=True)
        st.caption("Small standard deviations across folds mean the scores do not depend on one particular split.")
    with st.expander("What the metrics mean"):
        st.markdown(METRIC_HELP)

    c, d = st.columns(2)
    with c.container(border=True):
        plot(avp_chart())
    with d.container(border=True):
        plot(resid_hist_chart())
    e, f = st.columns(2)
    with e.container(border=True):
        plot(resid_pred_chart())
    with f.container(border=True):
        cal = BUNDLE["calibration"]
        lim = [float(min(cal["mean_pred"].min(), cal["mean_actual"].min())), float(max(cal["mean_pred"].max(), cal["mean_actual"].max()))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["mean_actual"], mode="lines+markers", name="Decile of predictions",
                                 line=dict(color=GREEN)))
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Ideal", line=dict(color=RED, dash="dash")))
        fig.update_layout(xaxis_title="Mean predicted (t/ha)", yaxis_title="Mean actual (t/ha)", legend=dict(orientation="h", y=1.1))
        plot(style(fig, 320, "Calibration by predicted-yield decile"))

    with st.container(border=True):
        gname = st.selectbox("Performance across groups (held-out test set)", list(BUNDLE["groups"].keys()), key="val_group")
        gt = BUNDLE["groups"][gname].copy()
        gt[["MAE", "RMSE", "R2", "Mean residual"]] = gt[["MAE", "RMSE", "R2", "Mean residual"]].round(3)
        st.dataframe(gt, hide_index=True, use_container_width=True)
        if (gt["n"] < 500).any():
            st.warning("Some groups have fewer than 500 test records; their scores are less certain.")
        spread = gt["R2"].max() - gt["R2"].min()
        st.caption(f"R² ranges from {gt['R2'].min():.3f} to {gt['R2'].max():.3f} across these groups (spread {spread:.3f}). "
                   "Large spreads would mean the model is unreliable for some conditions.")

    st.markdown("<div class='section-h'>Plausibility checks, data limitations and model limitations</div>", unsafe_allow_html=True)
    st.markdown(
        f"- **Implausible predictions:** {pl['pred_negative']:,} of {pl['n_test']:,} test predictions are negative and "
        f"{pl['pred_above_train_max']:,} exceed the highest training yield ({pl['train_yield_max']:.2f} t/ha). "
        f"Predicted range {pl['pred_min']:.2f}–{pl['pred_max']:.2f} t/ha.\n"
        f"- **Target quality:** {QR['n_negative']:,} negative yields (all at low rainfall) were excluded from fitting and primary test metrics; "
        "the sensitivity line above reports the effect.\n"
        f"- **Predictor structure:** the largest absolute correlation between any two numeric predictors is {QR['max_abs_predictor_corr']:.4f}, and the "
        f"most common category of each variable holds {', '.join(f'{k.replace(chr(95), chr(32))} {v:.0%}' for k, v in QR['cat_balance'].items())} of records. "
        "Field data is rarely this independent and balanced, which suggests the data may be simulated or curated; transfer to real farms is not established.\n"
        "- **Observational data:** the model learns associations; it cannot show that changing an input changes yield.\n"
        "- **Missing context:** no field identifiers, year, soil chemistry, seed variety or input quantities; fertilizer and irrigation are yes/no only.\n"
        f"- **Single seeded split:** test metrics come from one split (seed {cc.SEED}); cross-validation is the robustness check.")
    r2, sd = TM[BEST]["R2"], float(CV.set_index("Model").loc[BEST, "CV R2 sd"])
    if r2 > 0.5 and sd < 0.01 and pl["pred_negative"] == 0:
        st.success(f"Suitable for **exploratory decision support within this dataset** (test R² {r2:.2f}, CV R² sd {sd:.4f}, no negative predictions). "
                   "Not validated for real-world forecasting, causal claims or specific input recommendations.")
    else:
        st.warning(f"Use with caution (test R² {r2:.2f}, CV R² sd {sd:.4f}, {pl['pred_negative']:,} negative predictions). Treat outputs as rough exploration only.")


# =========================================================================== #
# PAGE: Methods & Data Quality
# =========================================================================== #
def analysis_summary_text() -> str:
    L = ["AgriYield Analytics: analysis summary", f"Dataset: {DATASET_NAME} (local file {Path(csv_path).name})",
         f"Rows: {QR['n_rows']:,} (read {QR['n_rows_read']:,}); columns: {QR['n_cols']}; valid records: {QR['n_valid']:,}",
         f"Missing cells in file: {QR['raw_missing']:,}; unusable rows removed: {QR['n_dropped_unusable']:,}; duplicates: {QR['n_duplicates']:,}",
         f"Negative yields: {QR['n_negative']:,} ({QR['pct_negative']:.4f}%); excluded from model fitting and primary test metrics",
         f"Split {int((1 - cc.TEST_SIZE) * 100)}/{int(cc.TEST_SIZE * 100)}, seed {cc.SEED}; train {META['n_train']:,}, test {META['n_test']:,}",
         f"Selected model (by CV RMSE): {BEST}", ""]
    L.append(model_table().to_string(index=False))
    s = BUNDLE["sensitivity_with_negatives"]
    L += ["", f"Sensitivity incl. negative yields (test): MAE {s['MAE']:.4f}, RMSE {s['RMSE']:.4f}, R2 {s['R2']:.4f}",
          "", "Limitations: observational data; associations only; no external validation; predictors unusually independent/balanced.",
          f"Generated: {META.get('trained_at', 'n/a')}"]
    return "\n".join(L)


def page_methods() -> None:
    st.markdown("### Methods & Data Quality")
    with st.container(border=True):
        st.markdown("**Dataset overview and schema**")
        roles = {"Region": "Predictor (category)", "Soil_Type": "Predictor (category)", "Crop": "Predictor (category)",
                 "Weather_Condition": "Predictor (category)", "Rainfall_mm": "Predictor (numeric)",
                 "Temperature_Celsius": "Predictor (numeric)", "Days_to_Harvest": "Predictor (numeric)",
                 "Fertilizer_Used": "Predictor (Boolean)", "Irrigation_Used": "Predictor (Boolean)", TARGET: "Target (numeric)"}
        schema = []
        for c_ in cc.ALL_COLUMNS:
            if c_ in CAT:
                rng = ", ".join(QR["categories"][c_])
            elif c_ in BOOL:
                rng = "True / False"
            else:
                s_ = QR["numeric_summary"].loc[c_]
                rng = f"{s_['min']:.2f} to {s_['max']:.2f}"
            schema.append({"Column": c_, "Role": roles[c_], "Type": str(DF[c_].dtype), "Observed values / range": rng})
        st.dataframe(pd.DataFrame(schema), hide_index=True, use_container_width=True)

    with st.container(border=True):
        st.markdown("**Data-quality checks (recomputed from the CSV)**")
        q = st.columns(5)
        q[0].metric("Rows × columns", f"{QR['n_rows']:,} × {QR['n_cols']}")
        q[1].metric("Empty cells", f"{QR['raw_missing']:,}")
        q[2].metric("Duplicate rows", f"{QR['n_duplicates']:,}")
        q[3].metric("Non-finite numbers", f"{QR['n_nonfinite']:,}")
        q[4].metric("Negative yields", f"{QR['n_negative']:,}")
        st.markdown("**Category consistency:** " + " · ".join(f"{c_.replace('_', ' ')}: {', '.join(v)}" for c_, v in QR["categories"].items())
                    + ". Unexpected spellings or extra categories would appear here.")
        st.dataframe(QR["numeric_summary"].round(3), use_container_width=True)
        if QR["n_negative"]:
            st.markdown(
                f"**Negative and implausible yields.** {QR['n_negative']:,} records ({QR['pct_negative']:.4f}%) have negative yield "
                f"(min {QR['neg_yield_min']:.2f}, mean {QR['neg_yield_mean']:.2f} t/ha); {QR['n_zero']:,} are exactly zero. All negatives occur at rainfall ≤ "
                f"{QR['neg_rain_max']:.0f} mm (mean {QR['neg_rain_mean']:.0f} vs {QR['all_rain_mean']:.0f} mm overall), so they follow a pattern rather than random typos. "
                "**Rule applied:** negatives are kept and flagged in the exploratory pages, excluded from model fitting and primary test metrics, "
                "and a sensitivity evaluation including them is reported on the Model Validation page. They are not silently deleted or imputed.")
            c1, c2 = st.columns(2)
            c1.dataframe(QR["neg_by_rain_band"].round(4), hide_index=True, use_container_width=True)
            with c2:
                for c_, vc in QR["neg_by_cat"].items():
                    st.write(f"**{c_.replace('_', ' ')}:** " + ", ".join(f"{k} {v}" for k, v in vc.items()))

    with st.container(border=True):
        st.markdown(
            f"""**Workflow**
- **Preprocessing:** categorical variables are one-hot encoded (linear model) or ordinally encoded as native categories (gradient boosting); Booleans become 0/1; numeric variables are used unscaled. All encoders sit inside scikit-learn pipelines fitted on training data only, including inside every cross-validation fold.
- **Train–test split:** random 80/20 split of all rows, seed {cc.SEED}, performed before any fitting. Negative-yield rows are removed from the training and test sets used for primary metrics ({META['n_excluded_train']:,} train and {META['n_excluded_test']:,} test rows).
- **Models:** baseline (training mean), linear regression, HistGradientBoostingRegressor (early stopping on an inner validation split of the training data; stopped after {META['hgb_iterations']} iterations).
- **Model selection:** lowest {cc.CV_FOLDS}-fold cross-validated RMSE on training data. Test-set scores are never used to choose the model.
- **Metrics:** MAE, RMSE and R² (definitions on the Predictive Analytics page).
- **Computational sampling:** models are fitted and tested on all valid rows. Samples are used only for: scatter plots (4,000 points on EDA pages; {len(BUNDLE['plot_df']):,} test points on model plots), cross-validation ({META['cv_sample']:,} training rows), and permutation importance ({cc.IMPORTANCE_SAMPLE:,} test rows). Seed {cc.SEED} everywhere.
- **Software:** scikit-learn {META['sklearn']}; trained {META.get('trained_at', 'n/a')} in {META['train_seconds']:.0f} s. Trained models are cached in `.cache/` and reused until the CSV changes."""
        )
    with st.container(border=True):
        st.markdown(
            """**Descriptive, predictive and prescriptive analytics**
- **Descriptive** (Overview, EDA): summarises what is in this dataset: means, distributions, correlations, group comparisons.
- **Predictive** (Predictive Analytics, Model Validation): estimates yield for records the model has not seen, with measured error.
- **Prescriptive** (Scenario Simulator): compares model predictions for hypothetical conditions. It is a what-if tool, not a recommender: the data are observational, so it cannot show that changing an input will change yield, and no quantities or schedules are advised.

**Study limitations:** observational data without field-level context; fertilizer and irrigation recorded as yes/no only; unusually independent and balanced predictors suggest the data may be simulated or curated; no external validation.""")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download analysis summary (TXT)", analysis_summary_text(), "agriyield_analysis_summary.txt", "text/plain")
    d2.download_button("⬇ Download model metrics (CSV)", model_table().to_csv(index=False), "agriyield_model_metrics.csv", "text/csv")


{"Overview": page_overview, "Exploratory Data Analysis": page_eda, "Predictive Analytics": page_predictive,
 "Scenario Simulator": page_scenarios, "Model Validation": page_validation, "Methods & Data Quality": page_methods}[page]()
