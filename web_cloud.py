import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Reoperation Risk (Python-only)", page_icon="🩺", layout="wide")

ART_DIR = Path("cloud_artifacts")  # where train_py.py saves artifacts
IMPUTER_PATH = ART_DIR / "imputer.pkl"
SCALER_PATH = ART_DIR / "scaler.pkl"
GMM_PATH = ART_DIR / "gmm.pkl"
LOGIT_PATH = ART_DIR / "logit.pkl"
META_PATH = ART_DIR / "meta.json"

DR_COLS = [f"Drainage{i}" for i in range(1, 6)]

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing {META_PATH}. Please run train_py.py locally and upload artifacts.")
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)
    gmm = joblib.load(GMM_PATH)
    logit = joblib.load(LOGIT_PATH)
    return meta, imputer, scaler, gmm, logit


def compute_probs(imputer, scaler, gmm, d1, d2, d3, d4, d5):
    X = np.array([[d1, d2, d3, d4, d5]], dtype=float)
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    probs = gmm.predict_proba(X_scaled)[0]
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    probs = probs / probs.sum()
    return probs


def number_line(prob, cutoff):
    fig, ax = plt.subplots(figsize=(9, 1.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('risk', [(0, (1,1,1)), (1, (0.8,0,0))])
    gradient = np.linspace(0, 1, 1000).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0.25, 0.75])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel('Risk probability', labelpad=10)
    ax.axvline(cutoff, ymin=0.2, ymax=0.8, color='black', linestyle='--', linewidth=2)
    ax.text(cutoff, 0.9, f"Cutoff = {cutoff:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.plot([prob], [0.5], marker='o', color='red', markersize=9, markeredgecolor='black')
    ax.text(prob, 0.1, f"p = {prob:.3f}", ha='center', va='top', fontsize=10)
    ax.text(0.01, 0.92, 'Low risk', ha='left', va='top', fontsize=10, color='#2e7d32')
    ax.text(0.99, 0.92, 'High risk', ha='right', va='top', fontsize=10, color='#c62828')
    plt.tight_layout()
    return fig


st.title("Reoperation Risk Prediction")

# Load artifacts
try:
    meta, imputer, scaler, gmm, logit = load_artifacts()
    st.success("Artifacts loaded.")
except Exception as e:
    st.error(str(e))
    st.stop()

K = int(meta["K"])  # number of classes
base_k = int(meta["base_k"])  # excluded in logistic features
cutoff = float(meta.get("cutoff", 0.0980660970811073))
scale_info = meta.get("scale_info", {})
scale_podmax = bool(scale_info.get("POD1_5max_scaled", False))
podmax_scale_factor = scale_info.get("POD1_5max_scale_factor", None)

# Inputs section (single page layout)
st.markdown("### Input: POD1–POD5 (mL)")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    d1 = st.number_input("POD1", min_value=0.0, value=100.0, step=1.0)
with col2:
    d2 = st.number_input("POD2", min_value=0.0, value=100.0, step=1.0)
with col3:
    d3 = st.number_input("POD3", min_value=0.0, value=100.0, step=1.0)
with col4:
    d4 = st.number_input("POD4", min_value=0.0, value=100.0, step=1.0)
with col5:
    d5 = st.number_input("POD5", min_value=0.0, value=100.0, step=1.0)

predict = st.button("Predict Reoperation Risk", type="primary")

if predict:
    # 1) posterior probs via GMM
    probs = compute_probs(imputer, scaler, gmm, d1, d2, d3, d4, d5)

    # 2) POD1-5max (with same scaling rule)
    pod15max = float(np.nanmax([d1, d2, d3, d4, d5]))
    if scale_podmax and podmax_scale_factor not in (None, 0):
        pod_feat = pod15max / float(podmax_scale_factor)
    else:
        pod_feat = pod15max

    # 3) build logistic features: K-1 probs excl base class, + POD1_5max_feat
    prob_cols = [f"prob{i}" for i in range(1, K+1)]
    base_name = f"prob{base_k}"
    feats = []
    for name in prob_cols:
        if name == base_name:
            continue
        idx = int(name.replace("prob", "")) - 1
        feats.append(probs[idx])
    feats.append(pod_feat)
    X_logit = np.array(feats, dtype=float).reshape(1, -1)

    # 4) predict
    p = float(logit.predict_proba(X_logit)[0, 1])
    p = float(np.clip(p, 1e-12, 1-1e-12))

    # Layout for outputs
    st.markdown("---")
    top1, top2 = st.columns([1, 1])

    with top1:
        st.subheader("Prediction")
        st.metric("Reoperation probability", f"{p:.2%}")
        risk_label = "High risk" if p >= cutoff else "Low risk"
        risk_color = "#c62828" if p >= cutoff else "#2e7d32"
        st.markdown(
            f"""
            <div style='display:inline-block; padding:6px 12px; border-radius:16px; background:{risk_color}1A; color:{risk_color}; font-weight:600;'>
                {risk_label}
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Number line
        fig = number_line(p, cutoff)
        st.pyplot(fig)

    with top2:
        st.subheader("Posterior class probabilities")
        # Beautified horizontal bar chart using Plotly
        df_probs = pd.DataFrame({"Class": [f"prob{i}" for i in range(1, K+1)], "Probability": probs})
        df_probs = df_probs.sort_values("Probability", ascending=True)
        fig_bar = px.bar(
            df_probs,
            x="Probability",
            y="Class",
            orientation="h",
            color="Probability",
            color_continuous_scale="Reds",
            range_x=[0, 1],
            height=350,
        )
        fig_bar.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            xaxis=dict(title="", tickformat=".0%"),
            yaxis=dict(title=""),
            template="simple_white",
        )
        # Add value labels
        fig_bar.update_traces(text=df_probs["Probability"].map(lambda v: f"{v:.2%}"), textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    with st.expander("Details (meta)"):
        st.json(meta)
