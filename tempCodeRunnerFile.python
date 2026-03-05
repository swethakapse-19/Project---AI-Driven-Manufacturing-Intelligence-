import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(
    page_title="AI Batch Optimization",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    background-color: #0b0e1a !important;
    color: #cdd6f4 !important;
}
.main .block-container {
    padding: 1.8rem 2.5rem 3rem 2.5rem;
    background-color: #0b0e1a;
    max-width: 1420px;
}

/* ── sidebar shell ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111425 0%, #0d1020 100%) !important;
    border-right: 1.5px solid #252b45 !important;
}

/* ── sidebar ALL text → medium-light blue-grey ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] [class*="sliderLabel"],
section[data-testid="stSidebar"] [class*="caption"],
section[data-testid="stSidebar"] small {
    color: #b0c4de !important;
    font-weight: 500 !important;
}

/* ── slider param headings specifically ── */
.param-label {
    font-size: 0.88rem;
    font-weight: 600;
    color: #aac4e0;
    letter-spacing: 0.4px;
    margin: 10px 0 2px 0;
    padding-left: 2px;
}

/* ── slider thumb & track ── */
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #38bdf8 !important;
    border: 2px solid #38bdf8 !important;
    box-shadow: 0 0 6px #38bdf870 !important;
    width: 18px !important;
    height: 18px !important;
}
section[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #38bdf8 !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    background: #111425 !important;
    border: 1px solid #38bdf840 !important;
    border-radius: 6px !important;
    padding: 2px 6px !important;
}

/* ── sidebar button ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%) !important;
    color: #fff !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(56,189,248,0.28) !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    box-shadow: 0 6px 26px rgba(56,189,248,0.45) !important;
    transform: translateY(-2px) !important;
}

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(145deg, #141927 0%, #1a2038 100%);
    border: 1px solid #252b45;
    border-radius: 16px;
    padding: 22px 14px 18px 14px;
    text-align: center;
    box-shadow: 0 2px 20px rgba(0,0,0,0.35);
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(56,189,248,0.14);
}
.kpi-icon  { font-size: 1.5rem; margin-bottom: 8px; }
.kpi-value { font-size: 2rem; font-weight: 800; line-height: 1.1; margin-bottom: 6px; }
.kpi-label {
    font-size: 0.7rem; font-weight: 600; color: #6b7a99;
    text-transform: uppercase; letter-spacing: 1.3px;
}

/* ── section divider header ── */
.sec-hdr {
    font-size: 0.78rem; font-weight: 700; color: #38bdf8;
    text-transform: uppercase; letter-spacing: 2px;
    border-left: 3px solid #38bdf8; padding-left: 10px;
    margin: 28px 0 14px 0;
}

/* ── result card ── */
.res-card {
    background: linear-gradient(145deg, #131726, #1b2240);
    border: 1px solid #252b45;
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.res-value { font-size: 1.65rem; font-weight: 800; margin-bottom: 4px; }
.res-label { font-size: 0.7rem; font-weight: 600; color: #6b7a99;
             text-transform: uppercase; letter-spacing: 1px; }

/* ── page hero card ── */
.hero {
    background: linear-gradient(135deg, #131726 0%, #1a2038 60%, #151e35 100%);
    border: 1px solid #252b45;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 6px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
}

/* ── expander ── */
div[data-testid="stExpander"] details summary {
    background: #131726 !important;
    border-radius: 10px !important;
    color: #38bdf8 !important;
    font-weight: 700 !important;
}

hr { border-color: #1e2540 !important; margin: 1.2rem 0; }

div[data-testid="stInfo"] {
    background: rgba(56,189,248,0.06) !important;
    border: 1px solid rgba(56,189,248,0.22) !important;
    border-radius: 12px !important;
    color: #8899aa !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    p  = pd.read_excel("Hackathon/_h_batch_process_data.xlsx")
    pr = pd.read_excel("Hackathon/_h_batch_production_data.xlsx")
    p["Batch_ID"]  = p["Batch_ID"].astype(str)
    pr["Batch_ID"] = pr["Batch_ID"].astype(str)
    df = pd.merge(p, pr, on="Batch_ID")

    df["Carbon_Emission"] = df["Power_Consumption_kW"] * 0.82

    feat = df[["Time_Minutes","Temperature_C","Pressure_Bar","Humidity_Percent",
               "Motor_Speed_RPM","Compression_Force_kN","Flow_Rate_LPM",
               "Power_Consumption_kW","Vibration_mm_s"]]
    tgt  = df[["Hardness","Dissolution_Rate","Content_Uniformity","Carbon_Emission"]]

    scaler = StandardScaler()
    X      = scaler.fit_transform(feat)
    Xtr,Xte,ytr,yte = train_test_split(X, tgt, test_size=0.2, random_state=42)

    mdl = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    mdl.fit(Xtr, ytr)
    pred = mdl.predict(Xte)

    df["Energy_Anomaly"]    = df["Power_Consumption_kW"] > df["Power_Consumption_kW"].mean() + 2*df["Power_Consumption_kW"].std()
    df["Carbon_Compliance"] = df["Carbon_Emission"] <= df["Carbon_Emission"].mean()

    return df, mdl, scaler, mean_absolute_error(yte,pred), r2_score(yte,pred)

def score(p): h,d,u,c=p; return h*.3+d*.3+u*.2-c*.2

def optimize(mdl, inp):
    bs,bc = -999,None
    for _ in range(100):
        v = inp + np.random.normal(0,.05,inp.shape)
        s = score(mdl.predict(v)[0])
        if s>bs: bs,bc=s,v
    return bc,bs

BG   = "#0b0e1a"
PBG  = "#0d1020"
AX   = "#3a4060"
TX   = "#a8b8d0"

with st.spinner("⚙️  Training AI model — please wait…"):
    df, model, scaler, mae, r2 = load_and_train()

# ─────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="display:flex; align-items:center; gap:14px;">
    <div style="font-size:2.8rem;">🏭</div>
    <div>
      <div style="font-size:2rem; font-weight:900; color:#e2e8f0; line-height:1.1;">
        AI Batch Optimization System
      </div>
      <div style="font-size:0.82rem; color:#6b7a99; margin-top:5px; letter-spacing:2px; font-weight:600;">
        SMART MANUFACTURING INTELLIGENCE DASHBOARD
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────
kpis = [
    ("📦", f"{df.shape[0]}",                          "Total Batches",        "#38bdf8"),
    ("📉", f"{mae:.4f}",                              "Model MAE",            "#818cf8"),
    ("📈", f"{r2:.4f}",                               "R² Score",             "#34d399"),
    ("⚠️", f"{int(df['Energy_Anomaly'].sum())}",      "Energy Anomalies",     "#f87171"),
    ("🌿", f"{df['Carbon_Compliance'].mean()*100:.1f}%","Carbon Compliance",  "#fbbf24"),
]
cols = st.columns(5)
for col,(icon,val,lbl,clr) in zip(cols,kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-value" style="color:{clr};">{val}</div>
          <div class="kpi-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# ANALYTICS CHARTS
# ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📊 Analytics Overview</div>', unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown(f"""
    <div style="font-size:0.83rem;font-weight:600;color:#8899aa;
                text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">
        ⚡ Energy Pattern — Anomaly Detection
    </div>""", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8,3.8))
    fig.patch.set_facecolor(PBG);  ax.set_facecolor("#0f1224")
    norm = df[~df["Energy_Anomaly"]];  anom = df[df["Energy_Anomaly"]]
    ax.fill_between(norm["Time_Minutes"], norm["Power_Consumption_kW"],
                    color="#38bdf8", alpha=0.12)
    ax.plot(norm["Time_Minutes"], norm["Power_Consumption_kW"],
            color="#38bdf8", lw=1.6, label="Normal")
    ax.scatter(anom["Time_Minutes"], anom["Power_Consumption_kW"],
               color="#f87171", s=52, zorder=6, edgecolors="#ff2255", lw=0.7,
               label=f"Anomaly  ({len(anom)})")
    ax.set_xlabel("Time (Minutes)", color=TX, fontsize=9, fontweight="600")
    ax.set_ylabel("Power (kW)",     color=TX, fontsize=9, fontweight="600")
    ax.tick_params(colors=AX, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e2540")
    ax.legend(facecolor="#111425", labelcolor=TX, edgecolor="#252b45", fontsize=8.5)
    ax.grid(axis="y", color="#1a2035", lw=0.6)
    st.pyplot(fig, use_container_width=True);  plt.close()

with right:
    st.markdown(f"""
    <div style="font-size:0.83rem;font-weight:600;color:#8899aa;
                text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">
        🔗 Correlation Matrix
    </div>""", unsafe_allow_html=True)

    cols4 = ["Power_Consumption_kW","Vibration_mm_s","Motor_Speed_RPM","Temperature_C"]
    corr  = df[cols4].corr()
    corr.index = corr.columns = ["Power","Vibr","RPM","Temp"]
    fig2, ax2 = plt.subplots(figsize=(5,3.8))
    fig2.patch.set_facecolor(PBG);  ax2.set_facecolor("#0f1224")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2,
                linewidths=1.2, linecolor="#0b0e1a",
                annot_kws={"size":10,"color":"white","fontweight":"bold"},
                cbar_kws={"shrink":0.82})
    ax2.tick_params(colors=TX, labelsize=8.5)
    ax2.set_xticklabels(ax2.get_xticklabels(), color=TX)
    ax2.set_yticklabels(ax2.get_yticklabels(), color=TX, rotation=0)
    st.pyplot(fig2, use_container_width=True);  plt.close()

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SIDEBAR  — parameter sliders
# ─────────────────────────────────────────────────────────
sb = st.sidebar

sb.markdown("""
<div style="background:linear-gradient(90deg,rgba(56,189,248,0.12),transparent);
            border-left:3px solid #38bdf8; padding:10px 14px; border-radius:0 8px 8px 0;
            margin-bottom:18px;">
  <span style="font-size:1rem;font-weight:800;color:#38bdf8;
               letter-spacing:1px;text-transform:uppercase;">🎛️ Process Parameters</span>
</div>
<p style="color:#6b7a99;font-size:0.78rem;margin:-8px 0 16px 0;">
  Adjust sliders then click <b style="color:#38bdf8;">PREDICT & OPTIMIZE</b>
</p>
""", unsafe_allow_html=True)

params = [
    ("⏱️  Time (Minutes)",          "time_v",  0.0, 200.0,  50.0),
    ("🌡️  Temperature (°C)",        "temp_v",  0.0, 100.0,  30.0),
    ("💨  Pressure (Bar)",          "pres_v",  0.0,  10.0,   1.0),
    ("💧  Humidity (%)",            "hum_v",   0.0, 100.0,  40.0),
    ("⚙️  Motor Speed (RPM)",       "spd_v",   0.0, 500.0, 100.0),
    ("🔩  Compression Force (kN)",  "frc_v",   0.0,  20.0,   3.0),
    ("🌊  Flow Rate (LPM)",         "flw_v",   0.0,  10.0,   1.5),
    ("⚡  Power Consumption (kW)",  "pwr_v",   0.0,  10.0,   2.0),
    ("📳  Vibration (mm/s)",        "vib_v",   0.0,  20.0,   3.0),
]

sv = {}
for lbl, key, mn, mx, dflt in params:
    sb.markdown(f'<div class="param-label">{lbl}</div>', unsafe_allow_html=True)
    sv[key] = sb.slider(" ", mn, mx, dflt, key=key, label_visibility="collapsed")

sb.markdown("<br>", unsafe_allow_html=True)
run_btn = sb.button("🚀  PREDICT & OPTIMIZE", use_container_width=True)

# ─────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔮 Prediction & Optimization</div>', unsafe_allow_html=True)

if run_btn:
    ud = np.array([[sv["time_v"], sv["temp_v"], sv["pres_v"], sv["hum_v"],
                    sv["spd_v"],  sv["frc_v"],  sv["flw_v"],  sv["pwr_v"], sv["vib_v"]]])
    us      = scaler.transform(ud)
    pred    = model.predict(us)[0]
    h,d,u,c = pred

    bc, bs  = optimize(model, us)
    opt     = model.predict(bc)[0]

    if "gold" not in st.session_state or bs > st.session_state.gold:
        st.session_state.gold = bs;  updated = True
    else:
        updated = False

    fields = [
        ("Hardness",          h,    opt[0], "#38bdf8"),
        ("Dissolution Rate",  d,    opt[1], "#818cf8"),
        ("Content Uniformity",u,    opt[2], "#34d399"),
        ("Carbon Emission",   c,    opt[3], "#f87171"),
    ]

    cp, co = st.columns(2, gap="large")

    with cp:
        st.markdown("""
        <div style="background:linear-gradient(145deg,#111726,#161f38);
                    border:1px solid #252b45;border-radius:14px;padding:18px 20px 10px;">
          <div style="font-size:0.72rem;font-weight:700;color:#38bdf8;
                      text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;">
            🔍 Current Prediction
          </div>
        """, unsafe_allow_html=True)
        for nm,v,_,clr in fields:
            st.markdown(f"""
            <div class="res-card">
              <div class="res-value" style="color:{clr};">{v:.3f}</div>
              <div class="res-label">{nm}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with co:
        st.markdown("""
        <div style="background:linear-gradient(145deg,#111726,#161f38);
                    border:1px solid #252b45;border-radius:14px;padding:18px 20px 10px;">
          <div style="font-size:0.72rem;font-weight:700;color:#34d399;
                      text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;">
            ✅ Optimized Output
          </div>
        """, unsafe_allow_html=True)
        for nm,_,ov,clr in fields:
            st.markdown(f"""
            <div class="res-card">
              <div class="res-value" style="color:{clr};">{ov:.3f}</div>
              <div class="res-label">Optimized {nm}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Bar chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.78rem;font-weight:600;color:#8899aa;
                text-transform:uppercase;letter-spacing:1.3px;margin-bottom:8px;">
        📊 Before vs After Optimization</div>""", unsafe_allow_html=True)

    labels = ["Hardness","Dissolution","Uniformity","Carbon"]
    bv = [h,d,u,c];  av = list(opt)
    xs = np.arange(len(labels));  w = 0.35

    fig3, ax3 = plt.subplots(figsize=(9,3.5))
    fig3.patch.set_facecolor(PBG);  ax3.set_facecolor("#0f1224")
    b1 = ax3.bar(xs-w/2, bv, w, color="#38bdf8", alpha=0.88, label="Current",  zorder=3)
    b2 = ax3.bar(xs+w/2, av, w, color="#34d399", alpha=0.88, label="Optimized", zorder=3)
    for bar in list(b1)+list(b2):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.008,
                 f"{bar.get_height():.2f}", ha="center", va="bottom",
                 color=TX, fontsize=8.5, fontweight="bold")
    ax3.set_xticks(xs);  ax3.set_xticklabels(labels, color=TX, fontweight="600", fontsize=10)
    ax3.tick_params(colors=AX, labelsize=8.5)
    ax3.grid(axis="y", color="#1a2035", lw=0.6, zorder=0)
    for sp in ax3.spines.values(): sp.set_edgecolor("#1e2540")
    ax3.legend(facecolor="#111425", labelcolor=TX, edgecolor="#252b45", fontsize=9.5)
    st.pyplot(fig3, use_container_width=True);  plt.close()

    # ── Score + Golden Signature
    st.markdown("<br>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2, gap="large")
    with sc1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-icon">🏆</div>
          <div class="kpi-value" style="color:#fbbf24;">{bs:.4f}</div>
          <div class="kpi-label">Optimization Score</div>
        </div>""", unsafe_allow_html=True)
    with sc2:
        gc  = "#34d399" if updated else "#fbbf24"
        glb = "⭐ Golden Signature — Updated!" if updated else "⭐ Golden Signature — Best This Session"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-icon">🥇</div>
          <div class="kpi-value" style="color:{gc};">{st.session_state.gold:.4f}</div>
          <div class="kpi-label">{glb}</div>
        </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:rgba(56,189,248,0.05); border:1.5px dashed rgba(56,189,248,0.25);
                border-radius:14px; padding:38px; text-align:center;">
      <div style="font-size:2rem; margin-bottom:10px;">🎯</div>
      <div style="color:#6b7a99; font-size:0.95rem; font-weight:500;">
        Adjust the sliders in the sidebar and click
        <span style="color:#38bdf8; font-weight:700;">PREDICT &amp; OPTIMIZE</span>
        to see AI-powered results
      </div>
    </div>""", unsafe_allow_html=True)

# ── Raw data
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋  View Raw Dataset"):
    st.dataframe(df, use_container_width=True, height=300)
