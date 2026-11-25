"""Fraud / Transaction Risk Detection: risk review console (Streamlit demo).

Run:  streamlit run app.py

The model only ranks transactions by risk and explains why. It never blocks or declines
anything: every alert is decided by a human analyst.
"""
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from src.explain import CATEGORY_NAMES
from src.service import RiskService
from src.threshold import FRICTION_COST, REVIEW_COST

st.set_page_config(page_title="Fraud / Transaction Risk Detection", page_icon="◎", layout="wide")

# palette: slate #1F2937 · navy #111827 · teal #0D9488 / #2DD4BF / #0F766E · white
RED, RED_SOFT, GREY_L, GREY_M = "#0D9488", "#2DD4BF", "#F1F5F9", "#64748B"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root { --slate:#1F2937; --navy:#111827; --teal:#0D9488; --teal-l:#2DD4BF; --teal-d:#0F766E; --white:#FFFFFF;
        --ink:#111827; --ink-2:#4B5563; --soft:#CBD5E1; --line-w:#E5E7EB; --line-d:#334155; }
.stApp, .stApp p, .stApp label, .stApp input, .stMarkdown { font-family:'Inter', sans-serif; }
.stApp { background: radial-gradient(1000px 420px at 100% -10%, rgba(45,212,191,.12), transparent 60%), var(--slate); }
[data-testid="stSidebar"] { background:var(--navy); border-right:1px solid var(--line-d); }
[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.6rem; max-width:1420px; }
[data-tag] { background:var(--teal) !important; }
[data-tag], [data-tag] * { color:#fff !important; fill:#fff !important; }

.brand { font-weight:800; letter-spacing:.32em; font-size:1.15rem; color:#fff; }
.brand span { color:var(--teal-l); }
.sub { color:var(--teal-l); font-size:.7rem; letter-spacing:.18em; text-transform:uppercase; margin-top:.2rem; }

.banner { display:flex; gap:.9rem; align-items:center; padding:.85rem 1.1rem; margin-bottom:1rem; background:var(--white);
          border-left:6px solid var(--teal); border-radius:12px; color:var(--ink); font-size:.9rem;
          box-shadow:0 6px 24px rgba(0,0,0,.18); }
.banner b { color:var(--teal-d); }
.banner .ic { font-size:1.35rem; color:var(--teal); }

.stTabs [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--line-d); }
.stTabs [data-baseweb="tab"] { background:transparent; border:none; padding:.6rem 1rem; color:var(--soft); }
.stTabs [aria-selected="true"] { color:#fff !important; box-shadow: inset 0 -3px 0 var(--teal-l); }
.stTabs [data-baseweb="tab-highlight"] { background:var(--teal-l); }

.h { font-size:1.2rem; font-weight:700; margin:.3rem 0 .7rem; color:#fff; }
.h small { color:var(--soft); font-weight:400; font-size:.82rem; margin-left:.45rem; }
.note { color:var(--soft); font-size:.86rem; line-height:1.55; }
.note b { color:#fff; }
.mono { font-family:'JetBrains Mono', monospace; }

.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:.7rem; margin:.4rem 0 1rem; }
.kpi { background:var(--white); border-radius:12px; padding:.85rem 1rem; box-shadow:0 4px 18px rgba(0,0,0,.15); }
.kpi .l { color:var(--ink-2); font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; font-weight:600; }
.kpi .v { font-size:1.65rem; font-weight:800; color:var(--ink); margin-top:.15rem; }
.kpi .d { color:var(--ink-2); font-size:.78rem; }
.kpi.alert { background:linear-gradient(135deg, var(--teal-d), var(--teal)); }
.kpi.alert .l, .kpi.alert .d { color:#CCFBF1; }
.kpi.alert .v { color:#fff; }

.alert-card { background:var(--white); border-left:6px solid var(--teal); border-radius:12px; padding:.9rem 1.05rem;
              margin-bottom:.4rem; box-shadow:0 4px 18px rgba(0,0,0,.15); }
.alert-top { display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; }
.amt { font-size:1.35rem; font-weight:800; color:var(--ink); }
.meta { color:var(--ink-2); font-size:.8rem; margin-top:.15rem; }
.score { text-align:right; }
.score .n { font-family:'JetBrains Mono', monospace; font-size:1.25rem; font-weight:700; color:var(--teal-d); }
.score .l { color:var(--ink-2); font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; }
.bar { height:7px; background:var(--line-w); border-radius:99px; overflow:hidden; margin:.55rem 0 .6rem; }
.bar > div { height:100%; background:linear-gradient(90deg, var(--teal-l), var(--teal-d)); border-radius:99px; }
.reason { font-size:.84rem; color:var(--ink); margin-top:.2rem; }
.reason::before { content:"▸ "; color:var(--teal); font-weight:700; }
.pill { display:inline-block; font-size:.67rem; letter-spacing:.08em; text-transform:uppercase; padding:.14rem .55rem;
        border-radius:99px; border:1px solid var(--line-w); color:var(--ink-2); margin-right:.3rem; font-weight:600; }
.pill.fraud { background:var(--navy); border-color:var(--navy); color:#fff; }
.pill.legit { background:#CCFBF1; border-color:#99F6E4; color:var(--teal-d); }
.pill.esc { border-color:var(--teal); color:var(--teal-d); }
.panel { background:var(--white); border-radius:12px; padding:1rem 1.1rem; box-shadow:0 4px 18px rgba(0,0,0,.15); }
.panel, .panel .note { color:var(--ink-2); }
.panel b, .panel .note b { color:var(--ink); }
.panel .meta { color:var(--ink-2); }
.panel .reason { color:var(--ink); }
.flow { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:.6rem; }
.step { background:var(--white); border-radius:12px; padding:.85rem; font-size:.84rem; color:var(--ink-2);
        border-top:4px solid var(--line-w); }
.step b { display:block; color:var(--ink); margin-bottom:.25rem; }
.step.human { border-top-color:var(--teal); background:#F0FDFA; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model, scores and explainer ...")
def get_service() -> RiskService:
    return RiskService()


@st.cache_data(show_spinner=False)
def busiest_day(threshold: float):
    svc = get_service()
    d = svc.test[svc.test["score"] >= threshold]
    return d["time"].dt.date.value_counts().idxmax()


@st.cache_data(show_spinner=False)
def curve(review: float, friction: float) -> pd.DataFrame:
    return get_service().cost_curve(review, friction)


@st.cache_data(show_spinner=False)
def importance() -> pd.Series:
    return get_service().global_importance()


def show_chart(chart, width="stretch"):
    """All charts share one look: clearly visible grid, readable axes, transparent background."""
    st.altair_chart(chart.configure_axis(grid=True, gridColor="#64748B", gridOpacity=0.55, gridDash=[3, 3],
                                         domainColor="#94A3B8", tickColor="#94A3B8", labelColor="#E5E7EB",
                                         titleColor="#CBD5E1", labelFontSize=11, titleFontSize=12)
                    .configure_view(strokeWidth=0), width=width)


svc = get_service()
if "decisions" not in st.session_state:
    st.session_state.decisions = {}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="brand">FRAUD <span>RISK</span></div><div class="sub">Transaction risk review</div>',
                unsafe_allow_html=True)
    st.caption("UneeQ Interns · ML Task: Fraud / Transaction Risk Detection · simulated card data (Kaggle)")
    st.divider()
    st.markdown("**Alert threshold**")
    threshold = st.slider("Risk score needed to send a transaction to review", 0.02, 0.99,
                          round(svc.recommended, 2), 0.01, label_visibility="collapsed")
    st.caption(f"Recommended: **{svc.recommended:.2f}** (lowest total cost on validation data)")
    st.markdown("**Business costs**")
    review_cost = st.number_input("Analyst review, $ per alert", 1.0, 200.0, REVIEW_COST, 1.0)
    friction_cost = st.number_input("Customer friction, $ per false alarm", 0.0, 200.0, FRICTION_COST, 1.0)
    st.caption("A missed fraud costs its full amount (charge-back).")

rec_m = svc.policy_metrics(svc.recommended)
RADAR_SVG = """
<svg class="radar" viewBox="0 0 340 230" xmlns="http://www.w3.org/2000/svg" aria-label="Animated risk radar">
  <defs>
    <radialGradient id="rg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#2DD4BF" stop-opacity=".28"/><stop offset="100%" stop-color="#2DD4BF" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="sw" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#2DD4BF" stop-opacity="0"/><stop offset="100%" stop-color="#2DD4BF" stop-opacity=".55"/>
    </linearGradient>
    <linearGradient id="cardg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0D9488"/><stop offset="100%" stop-color="#0F766E"/>
    </linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <!-- floating cards in the background -->
  <g opacity=".22" fill="#2DD4BF">
    <rect x="12" y="34" width="56" height="36" rx="6" transform="rotate(-12 40 52)"/>
    <rect x="270" y="20" width="56" height="36" rx="6" transform="rotate(10 298 38)"/>
    <rect x="18" y="160" width="56" height="36" rx="6" transform="rotate(8 46 178)"/>
    <rect x="268" y="168" width="56" height="36" rx="6" transform="rotate(-9 296 186)"/>
  </g>
  <circle cx="170" cy="115" r="105" fill="url(#rg)"/>
  <g fill="none" stroke="#2DD4BF" stroke-opacity=".35">
    <circle cx="170" cy="115" r="35"/><circle cx="170" cy="115" r="70"/><circle cx="170" cy="115" r="104"/>
    <line x1="66" y1="115" x2="274" y2="115"/><line x1="170" y1="11" x2="170" y2="219"/>
  </g>
  <!-- network lines -->
  <g stroke="#CCFBF1" stroke-opacity=".18">
    <line x1="112" y1="70" x2="150" y2="48"/><line x1="150" y1="48" x2="214" y2="62"/><line x1="214" y1="62" x2="238" y2="122"/>
    <line x1="238" y1="122" x2="206" y2="176"/><line x1="206" y1="176" x2="128" y2="172"/><line x1="128" y1="172" x2="98" y2="128"/>
    <line x1="98" y1="128" x2="112" y2="70"/>
  </g>
  <!-- rotating sweep -->
  <g class="sweep"><path d="M170 115 L274 115 A104 104 0 0 0 243.5 41.5 Z" fill="url(#sw)"/></g>
  <!-- ordinary transactions -->
  <g fill="#CBD5E1" fill-opacity=".65">
    <circle cx="112" cy="70" r="2.4"/><circle cx="150" cy="48" r="2.4"/><circle cx="214" cy="62" r="2.4"/>
    <circle cx="206" cy="176" r="2.4"/><circle cx="98" cy="128" r="2.4"/><circle cx="250" cy="90" r="2"/>
    <circle cx="90" cy="92" r="2"/><circle cx="190" cy="200" r="2"/><circle cx="140" cy="196" r="2"/>
  </g>
  <!-- alerts -->
  <g filter="url(#glow)" fill="#2DD4BF">
    <circle class="blip" cx="238" cy="122" r="4.5"/><circle class="blip b2" cx="128" cy="172" r="4.5"/>
    <circle class="blip b3" cx="226" cy="40" r="3.8"/>
  </g>
  <!-- shield + card -->
  <path d="M170 74 L200 86 V113 C200 133 187 147 170 154 C153 147 140 133 140 113 V86 Z" fill="#111827" fill-opacity=".85"
        stroke="#2DD4BF" stroke-width="2" filter="url(#glow)"/>
  <rect x="153" y="100" width="34" height="22" rx="3.5" fill="url(#cardg)" stroke="#CCFBF1" stroke-opacity=".6"/>
  <rect x="157" y="105" width="7" height="5.5" rx="1.2" fill="#F9FAFB" fill-opacity=".9"/>
  <rect x="157" y="115" width="22" height="2" rx="1" fill="#CCFBF1" fill-opacity=".7"/>
</svg>"""
st.markdown(f"""
<style>
.hero2 {{ display:grid; grid-template-columns:1.25fr 1fr; align-items:center; gap:1rem; margin-bottom:1rem;
         padding:1.2rem 1.5rem; border-radius:16px; border:1px solid rgba(45,212,191,.28);
         background:
           linear-gradient(rgba(45,212,191,.06) 1px, transparent 1px) 0 0/28px 28px,
           linear-gradient(90deg, rgba(45,212,191,.06) 1px, transparent 1px) 0 0/28px 28px,
           linear-gradient(135deg, #111827 0%, #13303a 60%, #0F766E 140%); overflow:hidden; }}
.hero2 h1 {{ margin:0; font-size:2.3rem; font-weight:800; letter-spacing:.14em; color:#fff; }}
.hero2 h1 span {{ color:#2DD4BF; }}
.hero2 .tag {{ color:#CBD5E1; font-size:.95rem; margin:.35rem 0 1rem; max-width:520px; line-height:1.5; }}
.hero2 .stats {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.55rem; max-width:560px; }}
.hero2 .stat {{ background:rgba(255,255,255,.95); border-radius:10px; padding:.5rem .7rem; }}
.hero2 .stat .v {{ font-weight:800; font-size:1.2rem; color:#111827; }}
.hero2 .stat .l {{ font-size:.66rem; letter-spacing:.1em; text-transform:uppercase; color:#4B5563; }}
.radar {{ width:100%; max-height:230px; }}
.radar .sweep {{ transform-box:view-box; transform-origin:170px 115px; animation:spin 4.5s linear infinite; }}
.radar .blip {{ transform-box:fill-box; transform-origin:center; animation:pulse 1.8s ease-in-out infinite; }}
.radar .b2 {{ animation-delay:.6s; }} .radar .b3 {{ animation-delay:1.2s; }}
@keyframes spin {{ to {{ transform:rotate(-360deg); }} }}
@keyframes pulse {{ 0%,100% {{ opacity:.55; transform:scale(.8); }} 50% {{ opacity:1; transform:scale(1.35); }} }}
@media (max-width: 900px) {{ .hero2 {{ grid-template-columns:1fr; }} }}
@media (prefers-reduced-motion: reduce) {{ .radar .sweep, .radar .blip {{ animation:none; }} }}
</style>
<div class="hero2">
  <div>
    <h1>FRAUD <span>RISK DETECTION</span></h1>
    <div class="tag">Spots suspicious card transactions, explains why, and sends them to a human analyst.
      Built for imbalanced data (1 fraud in ~250 payments) and real business costs.</div>
    <div class="stats">
      <div class="stat"><div class="v">{rec_m['recall']:.0%}</div><div class="l">fraud caught</div></div>
      <div class="stat"><div class="v">{rec_m['precision']:.0%}</div><div class="l">alerts are fraud</div></div>
      <div class="stat"><div class="v">{rec_m['alerts_per_day']:.0f}/day</div><div class="l">alerts to review</div></div>
      <div class="stat"><div class="v">${rec_m['savings'] / 1e6:.2f}M</div><div class="l">saved in 6 months</div></div>
    </div>
  </div>
  <div>{RADAR_SVG}</div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="banner"><span class="ic">⚠</span><div><b>Decision support only.</b> The model ranks '
            'transactions by risk and explains why. It <b>never blocks, declines or labels a transaction as fraud</b>: '
            'every alert is reviewed and decided by a human fraud analyst.</div></div>', unsafe_allow_html=True)

m = svc.policy_metrics(threshold, review_cost, friction_cost)
tab_q, tab_cost, tab_model, tab_explain, tab_how = st.tabs(
    ["◈  Review queue", "$  Threshold & cost", "▤  Model & imbalance", "?  Explain a score", "◎  How it works"])

# ------------------------------------------------------------------ review queue
with tab_q:
    left, right = st.columns([1, 3], gap="large")
    with left:
        dates = svc.dates()
        day = st.selectbox("Review day", dates, index=dates.index(busiest_day(round(svc.recommended, 2))),
                           format_func=lambda d: d.strftime("%a %d %b %Y"))
        q = svc.queue(day, threshold)
        total = int(q["day_total"].iloc[0]) if len(q) else int((svc.test["time"].dt.date == day).sum())
        decided = sum(1 for t in q["trans_num"] if t in st.session_state.decisions)
        st.markdown(f'<div class="kpis" style="grid-template-columns:1fr 1fr">'
                    f'<div class="kpi alert"><div class="l">Alerts</div><div class="v">{len(q)}</div>'
                    f'<div class="d">of {total:,} transactions</div></div>'
                    f'<div class="kpi"><div class="l">Reviewed</div><div class="v">{decided}/{len(q)}</div>'
                    f'<div class="d">by you today</div></div></div>', unsafe_allow_html=True)
        reveal = st.toggle("Reveal true label", value=True,
                           help="Demo only: real analysts don't know the answer in advance.")
        st.markdown('<div class="note">Each alert shows the <b>risk score</b> and the <b>top reasons</b> behind it. '
                    'You decide: <b>Confirm fraud</b> (contact customer, dispute), <b>Mark legit</b> (release), '
                    'or <b>Escalate</b> (senior review).</div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="h">Alert queue<small>{day.strftime("%A %d %B %Y")} · score ≥ {threshold:.2f} · '
                    f'riskiest first</small></div>', unsafe_allow_html=True)
        if q.empty:
            st.info("No transactions above the threshold on this day.")
        for _, r in q.iterrows():
            tid = r["trans_num"]
            dec = st.session_state.decisions.get(tid)
            reasons = "".join(f'<div class="reason">{escape(x["text"])}</div>' for x in r["reasons"])
            status = ""
            if dec:
                cls = {"Confirmed fraud": "fraud", "Marked legit": "legit", "Escalated": "esc"}[dec]
                status = f'<span class="pill {cls}">{dec}</span>'
                if reveal:
                    truth = "fraud" if r["is_fraud"] else "legit"
                    status += f'<span class="pill">Actual: {truth}</span>'
            st.markdown(
                f'<div class="alert-card"><div class="alert-top"><div><div class="amt">${r["amt"]:,.2f}</div>'
                f'<div class="meta">{escape(str(r["category_name"]))} · {r["time"].strftime("%H:%M")} · '
                f'{escape(str(r["city"]))}, {r["state"]} · card ••••{str(r["cc_num"])[-4:]} · '
                f'<span class="mono">{tid[:8]}</span></div></div>'
                f'<div class="score"><div class="n">{r["score"]:.3f}</div><div class="l">risk score</div></div></div>'
                f'<div class="bar"><div style="width:{r["score"] * 100:.0f}%"></div></div>{reasons}'
                f'<div style="margin-top:.5rem">{status}</div></div>', unsafe_allow_html=True)
            if not dec:
                b1, b2, b3, _ = st.columns([1.4, 1.1, 1, 1.6])
                if b1.button("Confirm fraud", key=f"f_{tid}", type="primary"):
                    st.session_state.decisions[tid] = "Confirmed fraud"; st.rerun()
                if b2.button("Mark legit", key=f"l_{tid}"):
                    st.session_state.decisions[tid] = "Marked legit"; st.rerun()
                if b3.button("Escalate", key=f"e_{tid}"):
                    st.session_state.decisions[tid] = "Escalated"; st.rerun()

# ------------------------------------------------------------------ threshold & cost
with tab_cost:
    st.markdown(f'<div class="h">Test period: {svc.days} days<small>Jun–Dec 2020 · {int(m["TP"] + m["FN"]):,} frauds '
                f'worth ${m["no_model_cost"]:,.0f}</small></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpis">
      <div class="kpi"><div class="l">Alerts per day</div><div class="v">{m['alerts_per_day']:.1f}</div>
        <div class="d">analyst workload</div></div>
      <div class="kpi"><div class="l">Precision</div><div class="v">{m['precision']:.0%}</div>
        <div class="d">of alerts are real fraud</div></div>
      <div class="kpi"><div class="l">Recall</div><div class="v">{m['recall']:.0%}</div>
        <div class="d">of fraud cases caught</div></div>
      <div class="kpi"><div class="l">Fraud $ caught</div><div class="v">${m['fraud_$_caught'] / 1e6:.2f}M</div>
        <div class="d">${m['fraud_$_missed']:,.0f} missed</div></div>
      <div class="kpi alert"><div class="l">Total cost</div><div class="v">${m['total_cost'] / 1e3:,.0f}K</div>
        <div class="d">saves ${m['savings'] / 1e6:.2f}M vs no model</div></div>
    </div>""", unsafe_allow_html=True)

    cv = curve(review_cost, friction_cost)
    best_t = svc.best_threshold(review_cost, friction_cost)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="h">Total cost vs threshold<small>reviews + friction + missed fraud</small></div>',
                    unsafe_allow_html=True)
        base = alt.Chart(cv).encode(x=alt.X("threshold:Q", title="Alert threshold", scale=alt.Scale(domain=[0, 1])))
        line = base.mark_line(color=GREY_L, strokeWidth=2.5).encode(
            y=alt.Y("total_cost:Q", title="Total cost ($)", scale=alt.Scale(type="log"), axis=alt.Axis(format="$~s")),
            tooltip=[alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("total_cost:Q", format="$,.0f"),
                     alt.Tooltip("alerts_per_day:Q", format=".1f", title="alerts/day")])
        rules = alt.Chart(pd.DataFrame({"x": [threshold, best_t], "who": ["Your threshold", "Lowest cost (validation)"]})
                          ).mark_rule(strokeWidth=2, strokeDash=[5, 4]).encode(
            x="x:Q", color=alt.Color("who:N", scale=alt.Scale(domain=["Your threshold", "Lowest cost (validation)"],
                                                              range=[RED_SOFT, GREY_M]),
                                     legend=alt.Legend(orient="top", title=None)))
        show_chart((line + rules).properties(height=290), width="stretch")
    with c2:
        st.markdown('<div class="h">Precision vs recall trade-off<small>move the slider in the sidebar</small></div>',
                    unsafe_allow_html=True)
        long = cv.melt("threshold", ["precision", "recall"], var_name="metric")
        pr = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
            x=alt.X("threshold:Q", title="Alert threshold", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["precision", "recall"], range=[RED_SOFT, GREY_L]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["metric", alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("value:Q", format=".1%")])
        mark = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color=RED_SOFT, strokeDash=[5, 4]).encode(x="x:Q")
        show_chart((pr + mark).properties(height=290), width="stretch")

    st.markdown('<div class="h">Policy comparison<small>thresholds chosen on validation (Mar–Jun 2020), '
                'results on test</small></div>', unsafe_allow_html=True)
    tc = svc.threshold_comparison()
    show = tc[["Policy", "threshold", "alerts_per_day", "precision", "recall", "fraud_$_caught", "fraud_$_missed",
               "total_cost", "savings_vs_no_model"]]
    st.dataframe(show.style.format({"threshold": "{:.2f}", "alerts_per_day": "{:.1f}", "precision": "{:.1%}",
                                    "recall": "{:.1%}", "fraud_$_caught": "${:,.0f}", "fraud_$_missed": "${:,.0f}",
                                    "total_cost": "${:,.0f}", "savings_vs_no_model": "${:,.0f}"}, na_rep="—"),
                 hide_index=True, width="stretch")
    st.markdown('<div class="note">Why not maximise F1? F1 treats a missed $900 fraud and a false alarm on a $5 coffee '
                'as equally bad. The cost view weighs them by money. Why not fill the team\'s capacity? Extra alerts are '
                'mostly false alarms: precision collapses and costs rise.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ model & imbalance
with tab_model:
    comp = svc.model_comparison()
    trap = (svc.test["is_fraud"] == 0).mean()
    st.markdown(f"""
    <div class="kpis">
      <div class="kpi"><div class="l">Fraud rate (test)</div><div class="v">{svc.test['is_fraud'].mean():.2%}</div>
        <div class="d">1 in {1 / svc.test['is_fraud'].mean():,.0f} transactions</div></div>
      <div class="kpi alert"><div class="l">The accuracy trap</div><div class="v">{trap:.1%}</div>
        <div class="d">accuracy of "always legit", which catches 0 fraud</div></div>
      <div class="kpi"><div class="l">Model PR-AUC (test)</div><div class="v">{comp['test_PR_AUC'].iloc[2:].max():.3f}</div>
        <div class="d">vs {comp['test_PR_AUC'].iloc[0]:.3f} random · {comp['test_PR_AUC'].iloc[1]:.3f} amount rule</div></div>
    </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown('<div class="h">Models & imbalance strategies<small>PR-AUC: area under the precision-recall '
                    'curve, higher is better</small></div>', unsafe_allow_html=True)
        bars = alt.Chart(comp).mark_bar(cornerRadiusEnd=3, height=20).encode(
            y=alt.Y("Model:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("test_PR_AUC:Q", title="Test PR-AUC", scale=alt.Scale(domain=[0, 1.1])),
            color=alt.condition(alt.datum.test_PR_AUC > 0.9, alt.value(RED_SOFT), alt.value(GREY_M)),
            tooltip=["Model", "Imbalance", alt.Tooltip("val_PR_AUC:Q", format=".3f"),
                     alt.Tooltip("test_PR_AUC:Q", format=".3f"), alt.Tooltip("test_ROC_AUC:Q", format=".4f"),
                     "train_seconds"])
        text = bars.mark_text(align="left", dx=5, color="#F9FAFB").encode(text=alt.Text("test_PR_AUC:Q", format=".3f"))
        show_chart((bars + text).properties(height=260), width="stretch")
    with c2:
        st.markdown('<div class="h">Why these metrics?</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel note">'
                    '<b>Accuracy</b> is misleading: predicting "legit" for everything is 99.6% accurate.<br><br>'
                    '<b>PR-AUC</b> measures how well fraud is ranked above legit, focusing on the rare fraud class. '
                    'It is the main model-selection metric.<br><br>'
                    '<b>Precision</b> = share of alerts that are real fraud (analyst time well spent).<br>'
                    '<b>Recall</b> = share of fraud caught (money protected).<br><br>'
                    '<b>Imbalance handling</b>: balanced class weights, undersampling (1 fraud : 10 legit) and no handling '
                    'rank fraud almost equally well. Weights mainly shift the score scale, so the <b>threshold is tuned '
                    'separately</b> on validation data. Undersampling trains much faster.</div>', unsafe_allow_html=True)
    st.dataframe(comp.style.format({"val_PR_AUC": "{:.3f}", "val_ROC_AUC": "{:.4f}", "test_PR_AUC": "{:.3f}",
                                    "test_ROC_AUC": "{:.4f}"}), hide_index=True, width="stretch")

# ------------------------------------------------------------------ explain
with tab_explain:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="h">What drives risk overall<small>mean |SHAP| over 6,000 test transactions</small></div>',
                    unsafe_allow_html=True)
        imp = importance().reset_index()
        imp.columns = ["Feature", "Importance"]
        show_chart(alt.Chart(imp).mark_bar(color=RED_SOFT, cornerRadiusEnd=3, height=14).encode(
            y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("Importance:Q", title="Average impact on the risk score"),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")]).properties(height=380), width="stretch")
        st.markdown('<div class="note">Gender is deliberately <b>not</b> a model input (protected attribute). '
                    'Customer age is used; see the fairness note in the evaluation report.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="h">What-if simulator<small>score a made-up transaction</small></div>',
                    unsafe_allow_html=True)
        cats = list(CATEGORY_NAMES)
        a1, a2 = st.columns(2)
        amt = a1.number_input("Amount ($)", 1.0, 5000.0, 950.0, 10.0)
        cat = a2.selectbox("Category", cats, format_func=CATEGORY_NAMES.get)
        hour = a1.slider("Hour of day", 0, 23, 2)
        ratio = a2.slider("× customer's usual amount", 0.1, 20.0, 6.0, 0.1)
        n24 = a1.slider("Purchases in previous 24h", 0, 15, 4)
        s24 = a2.number_input("Spend in previous 24h ($)", 0.0, 10000.0, 1500.0, 50.0)
        prev = a1.slider("Hours since previous purchase", 0.0, 48.0, 0.5, 0.25)
        hist = a2.slider("Customer's earlier transactions", 0, 3000, 600, 10)
        score, contrib, reasons = svc.what_if(amt, cat, hour, ratio, n24, s24, prev, cust_prior_txns=hist)
        verdict = "Sent to review queue" if score >= threshold else "Below threshold: no alert"
        st.markdown(f'<div class="panel"><div class="score" style="text-align:left"><div class="l">risk score</div>'
                    f'<div class="n" style="font-size:2rem">{score:.3f}</div></div>'
                    f'<div class="bar"><div style="width:{score * 100:.0f}%"></div></div>'
                    f'<div class="meta">{verdict} (threshold {threshold:.2f})</div>'
                    + "".join(f'<div class="reason">{escape(x["text"])}</div>' for x in reasons) + "</div>",
                    unsafe_allow_html=True)
        top = contrib.head(8).copy()
        top["Direction"] = top["Impact"].apply(lambda v: "towards fraud" if v > 0 else "towards legit")
        show_chart(alt.Chart(top).mark_bar(height=14, cornerRadiusEnd=3).encode(
            y=alt.Y("Feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X("Impact:Q", title="SHAP contribution (log-odds)"),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["towards fraud", "towards legit"],
                                                           range=[RED_SOFT, GREY_M]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Feature", alt.Tooltip("Impact:Q", format="+.2f")]).properties(height=230), width="stretch")

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown('<div class="h">Human-in-the-loop process</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="flow">
      <div class="step"><b>1 · Transaction</b>Card payment arrives (amount, merchant, time, location).</div>
      <div class="step"><b>2 · Features</b>Only past behaviour: usual spend, last-24h activity, time since previous purchase.</div>
      <div class="step"><b>3 · Risk score</b>Gradient-boosting model outputs a 0–1 score. No decision is made here.</div>
      <div class="step"><b>4 · Threshold</b>Above the cost-based threshold → alert. Below → payment proceeds normally.</div>
      <div class="step human"><b>5 · Analyst review 👤</b>A human sees the score and reasons and decides: confirm, release or escalate.</div>
      <div class="step"><b>6 · Feedback</b>Analyst outcomes become new labels for retraining and threshold reviews.</div>
    </div>""", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="panel note"><b>What the model does</b><br>• Ranks transactions by fraud risk<br>'
                    '• Explains each score with its top reasons (SHAP)<br>• Prioritises the analysts\' queue<br><br>'
                    '<b>What the model does NOT do</b><br>• Block, decline or reverse payments<br>'
                    '• Accuse a customer of fraud<br>• Replace the analyst\'s judgement</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel note"><b>Honest evaluation</b><br>• Time-based split: train 2019–Feb 2020, '
                    'validate Mar–Jun 2020, test Jun–Dec 2020<br>• Features use only past data; labels are never used as '
                    'features<br>• Thresholds chosen on validation, reported on test<br>• Data is <b>simulated</b> '
                    '(Sparkov generator), so real-world performance would be lower and needs a pilot.</div>',
                    unsafe_allow_html=True)
