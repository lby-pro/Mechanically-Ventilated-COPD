# -*- coding: utf-8 -*-
"""
FI-Lab and In-Hospital Mortality in Mechanically Ventilated COPD Patients
Streamlit prediction app — Multivariable Logistic Regression model.

Paper: "Association Between Laboratory Frailty Index and In-Hospital Mortality
        in Mechanically Ventilated COPD Patients: A Two-Cohort Study
        (MIMIC-IV and eICU-CRD)"

Model  : statsmodels Logit, 5 predictors, trained on NON-SCALED (raw) data
         predict interface: MODEL.predict(DataFrame[const, *FEATURES]) -> P(death)
Cohorts: MIMIC-IV (train 70% / test 30%) + eICU-CRD (external validation)
"""
import warnings
warnings.filterwarnings("ignore")

import io
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================
# CONFIG
# ================================================================
MODEL_FILE = "logistic_model.pkl"
ASSETS_FILE = "app_assets.pkl"

APP_TITLE = "FI-Lab and In-Hospital Mortality Risk in Mechanically Ventilated COPD Patients"
APP_SUBTITLE = ("Multivariable Logistic Regression · trained on MIMIC-IV, "
                "externally validated on eICU-CRD")
OUTCOME_LABEL = "In-Hospital Mortality"
RISK_TITLE = "Predicted Risk of In-Hospital Death"

SHAP_IMG_WIDTH = 720           # fixed pixel width — avoids full-container stretching
HIGH_THRESHOLD = 0.50
MID_THRESHOLD = 0.20

# ---- continuous / ordinal variables: label, unit, slider range, default, help ----
FEATURE_META = {
    "fi_lab_score": {
        "label": "Laboratory Frailty Index (FI-Lab)", "unit": "score",
        "lo": 0.00, "hi": 1.00, "step": 0.01, "default": 0.55,
        "help": "Laboratory Frailty Index = number of abnormal laboratory tests / total tests performed. "
                "Higher values indicate greater frailty. Cohort median ≈ 0.55.",
    },
    "sofa": {
        "label": "SOFA Score", "unit": "points",
        "lo": 0, "hi": 24, "step": 1, "default": 8,
        "help": "Sequential Organ Failure Assessment score. Cohort range 1–18, median 8.",
    },
    "oasis": {
        "label": "OASIS Score", "unit": "points",
        "lo": 0, "hi": 75, "step": 1, "default": 39,
        "help": "Oxford Acute Severity of Illness Score. Cohort range 22–58, median 39.",
    },
}

# ---- binary variables: option mapping, default, help ----
CATEGORICAL_META = {
    "cerebrovascular": {
        "label": "Cerebrovascular Disease",
        "options": {0: "No", 1: "Yes"}, "default": 0,
        "help": "Pre-existing cerebrovascular disease (ischaemic stroke / transient ischaemic attack).",
    },
    "CRRT": {
        "label": "Continuous Renal Replacement Therapy (CRRT)",
        "options": {0: "No", 1: "Yes"}, "default": 0,
        "help": "Whether CRRT was administered during the ICU stay.",
    },
}

# order in which the sidebar renders the inputs
SIDEBAR_ORDER = ["fi_lab_score", "sofa", "oasis", "cerebrovascular", "CRRT"]

# ================================================================
# Page setup & cached resources
# ================================================================
st.set_page_config(page_title="FI-Lab · COPD Mortality Risk",
                   page_icon="🫁", layout="wide", initial_sidebar_state="expanded")
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _find(name):
    """Locate a file next to app.py, falling back to the working directory."""
    p = os.path.join(APP_DIR, name)
    return p if os.path.exists(p) else name


@st.cache_resource
def load_model():
    with open(_find(MODEL_FILE), "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_assets():
    with open(_find(ASSETS_FILE), "rb") as f:
        return pickle.load(f)


MODEL = load_model()
ASSETS = load_assets()

FEATURE_ORDER = list(ASSETS["feature_order"])
RANGE_PARAMS = ASSETS["scaler_params"]
TRAIN_INFO = ASSETS["train_info"]
COEFS = ASSETS["coefficients"]
IS_STATSMODELS = ASSETS.get("model_type") == "statsmodels_logit"

LABELS = {**{k: m["label"] for k, m in FEATURE_META.items()},
          **{k: m["label"] for k, m in CATEGORICAL_META.items()}}


# ================================================================
# Inference — mirrors the training pipeline exactly
# ================================================================
def predict_probability(raw: dict) -> float:
    """raw user input -> probability of in-hospital death.

    This model was trained on NON-SCALED data: the raw values feed the linear
    predictor directly, and a leading `const` column carries the intercept.
    The `const` column is inserted by name so its position can never be wrong —
    statsmodels matches on position, so column order is critical.
    """
    X = pd.DataFrame([{v: float(raw[v]) for v in FEATURE_ORDER}], columns=FEATURE_ORDER)
    X.insert(0, "const", 1.0)
    if IS_STATSMODELS:
        out = MODEL.predict(X)                       # statsmodels .predict()
    else:
        out = MODEL.predict_proba(X)[:, 1]           # sklearn fallback
    return float(np.asarray(out).ravel()[0])


def linear_predictor(raw: dict) -> dict:
    """Transparent log-odds decomposition: contribution of each predictor."""
    parts = {"Intercept": float(COEFS.get("const", 0.0))}
    for v in FEATURE_ORDER:
        parts[v] = float(COEFS[v]) * float(raw[v])
    z = sum(parts.values())
    return {"parts": parts, "z": z}


# ================================================================
# SHAP
# ================================================================
@st.cache_resource
def get_shap_explainer():
    """Build the SHAP explainer over the 5 RAW predictors.

    The background matrix holds the 5 features only; the intercept column is added
    *inside* the model wrapper, so background and explained data always agree in shape.
    """
    import shap
    bg = pd.DataFrame(np.asarray(ASSETS["shap_background"]), columns=FEATURE_ORDER,
                      dtype=float).reset_index(drop=True)

    if IS_STATSMODELS:
        def f(X):
            Xd = pd.DataFrame(np.asarray(X, dtype=float), columns=FEATURE_ORDER)
            Xd.insert(0, "const", 1.0)
            return np.asarray(MODEL.predict(Xd)).ravel()
    else:
        def f(X):
            Xd = pd.DataFrame(np.asarray(X, dtype=float), columns=FEATURE_ORDER)
            return MODEL.predict_proba(Xd)[:, 1]

    return shap.KernelExplainer(f, bg)


def render_shap_waterfall(raw: dict):
    """Exact per-patient additive contribution waterfall, at a FIXED pixel width.

    For a logistic model the linear predictor decomposes exactly into

        z = intercept + Σ βᵢ·xᵢ ,   p = 1 / (1 + e^(−z))

    so each feature's contribution to the log-odds is simply ``βᵢ·xᵢ``. We plot that
    decomposition directly (starting from the intercept) instead of a
    background-relative SHAP attribution: a background of *typical* patients makes
    every bar ≈0 for a typical patient, which is correct but visually useless at the
    bedside. The zero-based form shows the true, interpretable magnitude of each
    driver, and the bar total reproduces the displayed probability exactly.

    (Verified against shap.LinearExplainer, which is exact for linear models — note
    shap cannot parse a statsmodels results object directly, hence the manual form.)
    """
    lp = linear_predictor(raw)
    z = lp["z"]

    # (key, display label, contribution) — keep the key so the caller can look up
    # anything by the original feature name after the bars have been re-sorted.
    triples = [(v, LABELS.get(v, v), lp["parts"][v]) for v in FEATURE_ORDER]
    active = [t for t in triples if abs(t[2]) > 1e-12]
    if not active:                       # every coefficient cancelled out
        active = [("__none__", "(no active drivers)", 0.0)]
    active.sort(key=lambda t: abs(t[2]))

    keys_sorted = [t[0] for t in active]
    names = [t[1] for t in active]
    contrib = np.array([t[2] for t in active], dtype=float)
    cumulative = np.cumsum(contrib)
    base = np.concatenate([[float(lp["parts"]["Intercept"])], cumulative[:-1]])
    colors = ["#d64545" if c > 0 else "#2f6fd0" for c in contrib]

    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(names) + 1.5), dpi=110)
    ax.barh(range(len(names)), contrib, left=base, color=colors, height=0.62,
            edgecolor="white", linewidth=0.8)

    span = max(abs(base.max() + contrib.max()), abs(base.min()), 1.0)
    pad = span * 0.055
    for i, (b, c) in enumerate(zip(base, contrib)):
        end = b + c
        ax.plot([b, end], [i + 0.42, i + 0.42], color="#888888", lw=0.8, ls=":", zorder=1)
        label = f"{c:+.3f}"
        ax.text(end + (pad if c >= 0 else -pad), i, label,
                va="center", ha="left" if c >= 0 else "right",
                fontsize=9, fontweight="bold",
                color="#8c2f2f" if c > 0 else "#24518f")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9.5)
    # mark the probability scale at the endpoints of the cumulative sum
    ax.set_xlabel("Contribution to log-odds  (red = increases risk, blue = decreases)",
                  fontsize=9.5)
    ax.set_title(f"Additive contribution of each feature to this patient's risk  "
                 f"(total z = {z:+.3f} → p = {1 / (1 + np.exp(-z)) * 100:.1f}%)",
                 fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", color="#e6e6e6", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(min(0.0, base.min()) - pad * 2.2, max(0.0, (base + contrib).max()) + pad * 2.2)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)                      # rewind — Streamlit reads from the current position
    # Centered fixed-width column so the PNG renders at a deterministic, bounded size
    # (avoids full-container stretching, which distorts the aspect ratio).
    pad_l, mid, pad_r = st.columns([0.03, 0.94, 0.03])
    with mid:
        st.image(buf, width=SHAP_IMG_WIDTH)

    # Return contributions keyed by ORIGINAL feature name, so the caller never has to
    # guess the ordering (the bars are sorted by |contribution| for display, which
    # would otherwise mis-pair values with feature keys).
    return {
        "contributions": {k: float(c) for k, c in zip(keys_sorted, contrib)},
        "intercept": float(lp["parts"]["Intercept"]),
        "z": z,
        "probability": float(1.0 / (1.0 + np.exp(-z))),
    }


# ================================================================
# Risk gauge
# ================================================================
def risk_gauge(prob: float):
    fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=120)
    left, theta, n_seg = np.pi * 0.75, np.pi * 1.5, 100
    thetas = np.linspace(left, left + theta, n_seg)
    for i, t in enumerate(thetas):
        frac = i / (n_seg - 1)
        color = "#2e9e5b" if frac < MID_THRESHOLD else ("#f2b134" if frac < HIGH_THRESHOLD else "#d64545")
        ax.plot([np.cos(t), np.cos(t + 0.016)], [np.sin(t), np.sin(t + 0.016)],
                lw=13, color=color, solid_capstyle="butt")
    angle = left + theta * min(max(prob, 0), 1)
    ax.annotate("", xy=(0.62 * np.cos(angle), 0.62 * np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2.5))
    ax.text(0, -0.18, f"{prob * 100:.1f}%", ha="center", va="center", fontsize=30,
            fontweight="bold",
            color="#d64545" if prob >= HIGH_THRESHOLD else ("#f2b134" if prob >= MID_THRESHOLD else "#2e9e5b"))
    ax.text(0, -0.45, RISK_TITLE, ha="center", va="center", fontsize=10, color="#555555")
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.55, 1.1); ax.axis("off")
    return fig


# ================================================================
# Model performance tab
# ================================================================
def page_performance():
    st.markdown("### 📈 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training set AUC (MIMIC-IV)", f"{TRAIN_INFO['auc_train']:.3f}",
              f"n = {TRAIN_INFO['n_train']}")
    c2.metric("Test set AUC (MIMIC-IV)", f"{TRAIN_INFO['auc_test']:.3f}",
              f"n = {TRAIN_INFO['n_test']}")
    c3.metric("External AUC (eICU-CRD)", f"{TRAIN_INFO['auc_ext']:.3f}",
              f"n = {TRAIN_INFO['n_ext']}")

    st.markdown("")
    rows = [
        {"Cohort": "MIMIC-IV · Training", "n": TRAIN_INFO["n_train"],
         "Event rate": f"{TRAIN_INFO['event_rate_train'] * 100:.1f}%",
         "AUC": f"{TRAIN_INFO['auc_train']:.3f}", "Accuracy": f"{TRAIN_INFO['acc_train']:.3f}"},
        {"Cohort": "MIMIC-IV · Test", "n": TRAIN_INFO["n_test"],
         "Event rate": f"{TRAIN_INFO['event_rate_test'] * 100:.1f}%",
         "AUC": f"{TRAIN_INFO['auc_test']:.3f}", "Accuracy": f"{TRAIN_INFO['acc_test']:.3f}"},
        {"Cohort": "eICU-CRD · External", "n": TRAIN_INFO["n_ext"],
         "Event rate": f"{TRAIN_INFO['event_rate_ext'] * 100:.1f}%",
         "AUC": f"{TRAIN_INFO['auc_ext']:.3f}", "Accuracy": f"{TRAIN_INFO['acc_ext']:.3f}"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("#### ROC Curves")
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=120)
    roc_t, roc_e = ASSETS.get("roc_test"), ASSETS.get("roc_external")
    if roc_t:
        ax.plot(roc_t["fpr"], roc_t["tpr"], lw=2, color="#1f77b4",
                label=f"MIMIC-IV test (AUC = {roc_t['auc']:.3f}, n = {roc_t['n_test']})")
    if roc_e:
        ax.plot(roc_e["fpr"], roc_e["tpr"], lw=2, color="#d62728", ls="--",
                label=f"eICU-CRD external (AUC = {roc_e['auc']:.3f}, n = {roc_e['n_test']})")
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1.4, label="Reference (random)")
    ax.set_xlabel("1 − Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("Discrimination of the Logistic Model", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### Model Coefficients (log-odds)")
    coef_rows = []
    for k, v in COEFS.items():
        if k == "const":
            continue
        coef_rows.append({"Feature": LABELS.get(k, k),
                          "β coefficient": f"{v:+.4f}",
                          "Odds ratio (OR)": f"{np.exp(v):.3f}"})
    st.dataframe(pd.DataFrame(coef_rows), hide_index=True, width="stretch")
    st.caption("Odds ratios are per one-unit increase in the predictor "
               "(or presence vs absence for binary variables), holding the others constant.")


# ================================================================
# Main prediction page
# ================================================================
def page_prediction():
    st.markdown(
        f"""
        <div style='background:linear-gradient(90deg,#1f4e79,#2d6da3);padding:22px 28px;
                    border-radius:14px;color:white;'>
            <h1 style='margin:0;font-size:24px;'>{APP_TITLE}</h1>
            <p style='margin:8px 0 0;font-size:14px;opacity:0.92;'>{APP_SUBTITLE}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("")

    with st.sidebar:
        st.markdown("### 📋 Patient Feature Input")
        raw = {}
        st.markdown("#### 🧪 Laboratory Frailty Index")
        for v in ["fi_lab_score"]:
            m = FEATURE_META[v]
            raw[v] = st.slider(f"{m['label']} ({m['unit']})", float(m["lo"]), float(m["hi"]),
                               float(m["default"]), step=float(m["step"]), help=m["help"])
        st.markdown("#### 🩺 Severity Scores")
        for v in ["sofa", "oasis"]:
            m = FEATURE_META[v]
            raw[v] = st.slider(f"{m['label']} ({m['unit']})", int(m["lo"]), int(m["hi"]),
                               int(m["default"]), step=int(m["step"]), help=m["help"])
        st.markdown("#### 🏷️ Clinical Conditions")
        for v in ["cerebrovascular", "CRRT"]:
            m = CATEGORICAL_META[v]
            options = list(m["options"].keys())
            labels = [m["options"][k] for k in options]
            sel = st.selectbox(m["label"], labels, index=options.index(m["default"]), help=m["help"])
            raw[v] = options[labels.index(sel)]
        st.divider()
        predict_clicked = st.button("🔮 Predict Risk", type="primary", width="stretch")

    col_info, col_result = st.columns([1.05, 1])

    with col_info:
        st.markdown("#### 📥 Entered Patient Features")
        rows = []
        for v in SIDEBAR_ORDER:
            if v in CATEGORICAL_META:
                val = CATEGORICAL_META[v]["options"][raw[v]]
            else:
                val = f"{raw[v]:g} {FEATURE_META[v]['unit']}"
            rows.append({"Feature": LABELS[v], "Entered Value": val})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.info("ℹ️ This model uses **raw, non-standardised** values — the logistic "
                "regression was trained on the original scale, so no unit conversion "
                "or z-scoring is applied to your inputs.", icon="ℹ️")

    if not predict_clicked:
        with col_result:
            st.markdown("#### 📤 Prediction Result")
            st.warning("Set the patient features in the sidebar, then click **🔮 Predict Risk**.")
        return

    prob = predict_probability(raw)

    with col_result:
        st.markdown("#### 📤 Prediction Result")
        c1, c2 = st.columns([1, 1.05])
        with c1:
            st.pyplot(risk_gauge(prob), clear_figure=True)
        with c2:
            st.metric(f"{OUTCOME_LABEL} Probability", f"{prob * 100:.1f}%",
                      delta=f"{'High risk' if prob >= HIGH_THRESHOLD else 'Not high risk'} "
                            f"(threshold {HIGH_THRESHOLD:.0%})",
                      delta_color="inverse")
            st.progress(min(max(prob, 0.0), 1.0))

        if prob >= HIGH_THRESHOLD:
            st.error("🔴 **High risk predicted.** Consider escalating monitoring, reassessing "
                     "organ support, and discussing prognosis and goals of care early.", icon="🚨")
        elif prob >= MID_THRESHOLD:
            st.warning("🟡 **Intermediate risk predicted.** Monitor closely, reassess "
                       "reversible organ dysfunction, and stay alert for deterioration.", icon="⚠️")
        else:
            st.success("🟢 **Low risk predicted.** Continue standard management with regular "
                       "re-evaluation.", icon="✅")

    st.divider()
    st.markdown("#### 🔍 Model Explainability")
    st.markdown("Red bars **increase** the predicted risk, blue bars **decrease** it; "
                "a longer bar means a larger contribution.")
    try:
        expl = render_shap_waterfall(raw)
        contribs = expl["contributions"]
        ranked = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))
        up = [k for k, v in ranked if v > 0][:3]
        down = [k for k, v in ranked if v < 0][:3]
        parts = []
        if up:
            parts.append("**Main factors increasing risk:** " + ", ".join(LABELS.get(v, v) for v in up))
        if down:
            parts.append("**Main factors decreasing risk:** " + ", ".join(LABELS.get(v, v) for v in down))
        if parts:
            st.markdown("　".join(parts))
    except Exception as e:  # noqa: BLE001
        st.warning(f"Explanation could not be rendered ({e}); the prediction above is unaffected.")

    with st.expander("🧮 Transparent log-odds decomposition"):
        lp = linear_predictor(raw)
        dec = pd.DataFrame(
            [{"Component": ("Intercept" if k == "Intercept" else LABELS.get(k, k)),
              "Contribution (log-odds)": f"{v:+.4f}"} for k, v in lp["parts"].items()]
        )
        dec.loc[len(dec)] = {"Component": "Linear predictor (sum)", "Contribution (log-odds)": f"{lp['z']:+.4f}"}
        st.dataframe(dec, hide_index=True, width="stretch")
        st.caption(f"Probability = 1 / (1 + e^(−{lp['z']:.4f})) = {prob * 100:.2f}%")

    st.caption("⚠️ Disclaimer: this tool is intended for research and educational purposes only. "
               "It was developed on retrospective ICU data and must not replace clinical judgement.")


def main():
    with st.sidebar:
        st.markdown("---")
        page = st.radio("Navigation", ["🫁 Risk Prediction", "📈 Model Performance"],
                        label_visibility="collapsed")

    if page.startswith("🫁"):
        page_prediction()
    else:
        page_performance()

    st.divider()
    st.caption("FI-Lab & COPD in-hospital mortality · Logistic regression · "
               "MIMIC-IV (development) + eICU-CRD (external validation) · For research use only.")


if __name__ == "__main__":
    main()
