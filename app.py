"""
app.py — Credit Card Fraud Detector (Streamlit)
Uses ann_model.h5 + scaler.pkl from model_scaler/ folder.
Pure numpy inference — NO TensorFlow required.

FIXES:
  1. Preset buttons now correctly apply values via session_state keys.
  2. All sliders/inputs are bound to session_state to survive reruns.
  3. Scaler handles both (2,) and (30,) fit shapes gracefully.
  4. H5 weight loader has a robust multi-path search so None weights
     don't silently crash ann_predict.
  5. ann_predict guards against None weight arrays with clear errors.
"""

import streamlit as st
import numpy as np
import joblib
import h5py
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1d27 100%); }

    .hero-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d47a1 50%, #1565c0 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(13, 71, 161, 0.4);
    }
    .hero-header h1 { color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero-header p  { color: #e3f2fd; font-size: 1rem; margin: 0.5rem 0 0 0; }

    .fraud-badge {
        background: linear-gradient(135deg, #b71c1c, #e53935);
        color: white; border-radius: 50px; padding: 1rem 2rem;
        font-size: 1.5rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 20px rgba(229,57,53,0.5);
        animation: pulse 1.5s infinite;
    }
    .legit-badge {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: white; border-radius: 50px; padding: 1rem 2rem;
        font-size: 1.5rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 20px rgba(46,125,50,0.5);
    }
    @keyframes pulse {
        0%   { box-shadow: 0 4px 20px rgba(229,57,53,0.5); }
        50%  { box-shadow: 0 4px 30px rgba(229,57,53,0.9); }
        100% { box-shadow: 0 4px 20px rgba(229,57,53,0.5); }
    }
    .section-header {
        font-size: 1rem; font-weight: 600; color: #90caf9;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin-bottom: 0.5rem; padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(144,202,249,0.2);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-section {
        background: rgba(255,255,255,0.04); border-radius: 10px;
        padding: 1rem; margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: white; border: none; border-radius: 10px;
        font-weight: 600; font-size: 1rem; padding: 0.7rem 2rem;
        width: 100%; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(21,101,192,0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(21,101,192,0.6);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PRESET DATA
# ─────────────────────────────────────────────────────────────────────────────
PRESET_LEGIT = {
    "Time": 406.0, "Amount": 149.62,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782, "V5": -0.3383,
    "V6": 0.4624, "V7": 0.2396, "V8": 0.0987, "V9": 0.3638, "V10": 0.0908,
    "V11": -0.5516, "V12": -0.6178, "V13": -0.9913, "V14": -0.3112,
    "V15": 1.4682, "V16": -0.4704, "V17": 0.2080, "V18": 0.0258,
    "V19": 0.4040, "V20": 0.2514, "V21": -0.0183, "V22": 0.2778,
    "V23": -0.1105, "V24": 0.0669, "V25": 0.1285, "V26": -0.1891,
    "V27": 0.1336, "V28": -0.0211,
}

PRESET_FRAUD = {
    "Time": 406.0, "Amount": 1.0,
    "V1": -3.0435, "V2": 3.9262, "V3": -4.4897, "V4": 3.4935, "V5": -3.1723,
    "V6": 0.4430, "V7": -3.2015, "V8": 0.6006, "V9": -1.3434, "V10": -2.2523,
    "V11": 1.9997, "V12": -4.2893, "V13": 0.3880, "V14": -3.5637,
    "V15": -0.3842, "V16": -0.9528, "V17": -3.0065, "V18": -1.7870,
    "V19": 0.1696, "V20": 0.4256, "V21": 0.5264, "V22": -0.0440,
    "V23": 0.4863, "V24": -0.0832, "V25": 0.3585, "V26": 0.0739,
    "V27": 0.3142, "V28": 0.2533,
}

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — Initialise session_state keys BEFORE widgets are rendered
# This ensures preset buttons can write values that sliders/inputs read.
# ─────────────────────────────────────────────────────────────────────────────
def _default(key):
    if key in ("Time",):   return 0.0
    if key == "Amount":    return 10.0
    return 0.0

for feat in FEATURE_ORDER:
    sk = f"inp_{feat}"
    if sk not in st.session_state:
        st.session_state[sk] = _default(feat)


# ─────────────────────────────────────────────────────────────────────────────
# PURE NUMPY ANN INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model_scaler")
MODEL_PATH  = os.path.join(MODEL_DIR, "ann_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
# import st
# st.write("BASE_DIR:", BASE_DIR)
# st.write("MODEL_PATH exists:", os.path.exists(MODEL_PATH), "→", MODEL_PATH)
# st.write("SCALER_PATH exists:", os.path.exists(SCALER_PATH), "→", SCALER_PATH)
# st.write("Files in model_scaler folder:", os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else "FOLDER NOT FOUND")


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def batch_norm_inference(x, gamma, beta, moving_mean, moving_var, eps=1e-3):
    x_norm = (x - moving_mean) / np.sqrt(moving_var + eps)
    return gamma * x_norm + beta


@st.cache_resource
def load_ann_weights(model_path: str):
    """
    FIX 2 — Robust multi-strategy weight loader.
    Tries nested path first, then flat model_weights search,
    so it works regardless of how Keras saved the HDF5.
    """
    weights = {}

    def _arr(group, *keys):
        """Walk a chain of keys; return numpy array or None."""
        cur = group
        for k in keys:
            if k not in cur:
                return None
            cur = cur[k]
        return np.array(cur)

    def find_layer(mw, layer_name):
        """
        Search for a layer's weight group using multiple strategies:
          1. mw[layer_name][model_name][layer_name]  (Keras functional)
          2. mw[layer_name][layer_name]               (some SavedModels)
          3. mw[layer_name]                           (flat layout)
        Returns the h5py group containing weight arrays, or None.
        """
        if layer_name not in mw:
            return None
        g = mw[layer_name]
        # strategy 1: mw / layer / AnyModelName / layer
        for model_key in g.keys():
            if layer_name in g[model_key]:
                return g[model_key][layer_name]
        # strategy 2: mw / layer / layer
        if layer_name in g:
            return g[layer_name]
        # strategy 3: mw / layer  (weights directly here)
        return g

    with h5py.File(model_path, "r") as f:
        mw = f["model_weights"]

        # ── Dense weights ──────────────────────────────────────────────────
        for layer_name, prefix in [
            ("hidden_1", "h1"), ("hidden_2", "h2"),
            ("hidden_3", "h3"), ("output",   "out"),
        ]:
            grp = find_layer(mw, layer_name)
            if grp is None:
                st.error(f"Weight layer '{layer_name}' not found in HDF5.")
                weights[f"{prefix}_kernel"] = None
                weights[f"{prefix}_bias"]   = None
            else:
                weights[f"{prefix}_kernel"] = np.array(grp["kernel"])
                weights[f"{prefix}_bias"]   = np.array(grp["bias"])

        # ── BatchNorm weights ──────────────────────────────────────────────
        for bn_layer, key in [
            ("batch_normalization",   "bn1"),
            ("batch_normalization_1", "bn2"),
        ]:
            grp = find_layer(mw, bn_layer)
            if grp is None:
                weights[f"{key}_gamma"] = None   # signals BN absent
            else:
                weights[f"{key}_gamma"]       = np.array(grp["gamma"])
                weights[f"{key}_beta"]        = np.array(grp["beta"])
                weights[f"{key}_moving_mean"] = np.array(grp["moving_mean"])
                weights[f"{key}_moving_var"]  = np.array(grp["moving_variance"])

    return weights


def ann_predict(x: np.ndarray, weights: dict) -> float:
    """
    FIX 3 — Guard against None weight arrays with actionable errors.
    x: shape (1, 30) — already scaled.
    Returns float probability of fraud (0–1).
    """
    for needed in ("h1_kernel", "h1_bias", "h2_kernel", "h2_bias",
                   "h3_kernel", "h3_bias", "out_kernel", "out_bias"):
        if weights.get(needed) is None:
            raise ValueError(
                f"Weight '{needed}' is None — check H5 layer names in your model file."
            )

    # Layer 1: Dense → BatchNorm → ReLU
    z1 = x @ weights["h1_kernel"] + weights["h1_bias"]
    if weights.get("bn1_gamma") is not None:
        z1 = batch_norm_inference(
            z1, weights["bn1_gamma"], weights["bn1_beta"],
            weights["bn1_moving_mean"], weights["bn1_moving_var"]
        )
    a1 = relu(z1)

    # Layer 2: Dense → BatchNorm → ReLU
    z2 = a1 @ weights["h2_kernel"] + weights["h2_bias"]
    if weights.get("bn2_gamma") is not None:
        z2 = batch_norm_inference(
            z2, weights["bn2_gamma"], weights["bn2_beta"],
            weights["bn2_moving_mean"], weights["bn2_moving_var"]
        )
    a2 = relu(z2)

    # Layer 3: Dense → ReLU
    z3 = a2 @ weights["h3_kernel"] + weights["h3_bias"]
    a3 = relu(z3)

    # Output: Dense → Sigmoid
    z4    = a3 @ weights["out_kernel"] + weights["out_bias"]
    prob  = sigmoid(z4)
    return float(prob.flatten()[0])


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_scaler(scaler_path: str):
    return joblib.load(scaler_path)


model_loaded  = False
scaler_loaded = False
ann_weights   = None
scaler        = None

if os.path.exists(MODEL_PATH):
    try:
        ann_weights  = load_ann_weights(MODEL_PATH)
        model_loaded = True
    except Exception as e:
        st.error(f"Failed to load model: {e}")

if os.path.exists(SCALER_PATH):
    try:
        scaler        = load_scaler(SCALER_PATH)
        scaler_loaded = True
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Fraud Detector")
    st.markdown("---")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🔬 About This App")
    st.markdown("""
Uses a **3-layer ANN** trained on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- **284,807** transactions  
- **492** fraud cases (0.17%)
- SMOTE for class balance
- Pure **NumPy** inference — no TensorFlow!
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Model Status")
    if model_loaded:
        st.success("✅ ANN Model Loaded")
    else:
        st.error(f"❌ Model not found at\n`{MODEL_PATH}`")
    if scaler_loaded:
        st.success("✅ Scaler Loaded")
    else:
        st.error(f"❌ Scaler not found at\n`{SCALER_PATH}`")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🏗️ ANN Architecture")
    st.markdown("""
| Layer | Size | Activation |
|-------|------|-----------|
| Input | 30 | — |
| Hidden 1 | 128 | ReLU |
| BatchNorm | 128 | — |
| Dropout | 40% | — |
| Hidden 2 | 64 | ReLU |
| BatchNorm | 64 | — |
| Dropout | 30% | — |
| Hidden 3 | 32 | ReLU |
| Dropout | 20% | — |
| Output | 1 | Sigmoid |
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 V1–V28** are PCA-transformed confidential transaction features.")


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>💳 Credit Card Fraud Detector</h1>
    <p>ANN-powered real-time fraud detection &nbsp;|&nbsp; Pure NumPy inference &nbsp;|&nbsp; No TensorFlow required</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded or not scaler_loaded:
    st.error("⚠️ Place `ann_model.h5` and `scaler.pkl` inside the `model_scaler/` folder to enable predictions.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — PRESET BUTTONS (must come BEFORE widgets so state is set first)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Quick Presets</p>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("🟢 Typical Legit Transaction"):
        for feat, val in PRESET_LEGIT.items():
            st.session_state[f"inp_{feat}"] = float(val)
        st.rerun()

with col_b:
    if st.button("🔴 Suspicious Transaction"):
        for feat, val in PRESET_FRAUD.items():
            st.session_state[f"inp_{feat}"] = float(val)
        st.rerun()

with col_c:
    if st.button("⚪ Reset to Zero"):
        for feat in FEATURE_ORDER:
            st.session_state[f"inp_{feat}"] = _default(feat)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION — widgets bound to session_state via key=
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Transaction Details</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["⚙️ Basic Info", "🔢 V1 – V14 (PCA Features)", "🔢 V15 – V28 (PCA Features)"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "⏱ Time (seconds since first transaction)",
            value=None,          # value comes from session_state
            step=1.0, format="%.2f",
            key="inp_Time",
            help="Seconds elapsed from dataset start (range: 0 – 172,792)"
        )
        st.number_input(
            "💰 Amount ($)",
            value=None,
            min_value=0.0, step=0.01, format="%.2f",
            key="inp_Amount",
            help="Transaction amount in USD"
        )
    with col2:
        st.info("""
**How to use:**
- **Time**: Seconds since dataset start  
- **Amount**: Transaction dollar value  
- **V1–V28**: PCA-transformed features (normal range: -3 to 3)

🔴 Fraud often shows extreme V-values + unusual amounts.
        """)

with tab2:
    cols = st.columns(3)
    for idx, i in enumerate(range(1, 15)):
        with cols[idx % 3]:
            st.slider(f"V{i}", -5.0, 5.0, step=0.01, key=f"inp_V{i}")

with tab3:
    cols = st.columns(3)
    for idx, i in enumerate(range(15, 29)):
        with cols[idx % 3]:
            st.slider(f"V{i}", -5.0, 5.0, step=0.01, key=f"inp_V{i}")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Analyze Transaction", use_container_width=True)

if predict_clicked:
    st.markdown("---")
    st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)

    # Read feature values from session_state (guaranteed to reflect current widget state)
    raw_input = np.array(
        [[st.session_state[f"inp_{f}"] for f in FEATURE_ORDER]],
        dtype=np.float64
    )  # shape (1, 30)

    # ── FIX 5 — Scaler: handle both fit-on-2 and fit-on-30 shapes ──────────
    raw_scaled = raw_input.copy()
    if scaler is not None:
        try:
            n_scaler_features = scaler.n_features_in_
            if n_scaler_features == 30:
                # Scaler was fit on all 30 features
                raw_scaled = scaler.transform(raw_input)
            elif n_scaler_features == 2:
                # Scaler was fit on [Time, Amount] only
                raw_scaled[:, [0, 29]] = scaler.transform(raw_input[:, [0, 29]])
            else:
                st.warning(
                    f"Unexpected scaler feature count ({n_scaler_features}). "
                    "Attempting transform on full vector."
                )
                raw_scaled = scaler.transform(raw_input)
        except Exception as e:
            st.warning(f"Scaler transform issue (using raw values): {e}")

    # ANN forward pass (pure numpy)
    try:
        fraud_prob = ann_predict(raw_scaled, ann_weights)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    legit_prob = 1.0 - fraud_prob
    prediction = 1 if fraud_prob >= 0.5 else 0

    # ── Result display ─────────────────────────────────────────────────────
    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        if prediction == 1:
            st.markdown(f"""
            <div class="fraud-badge">
                🚨 FRAUDULENT TRANSACTION<br>
                <small style="font-size:1rem;font-weight:400;">Confidence: {fraud_prob*100:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-badge">
                ✅ LEGITIMATE TRANSACTION<br>
                <small style="font-size:1rem;font-weight:400;">Confidence: {legit_prob*100:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)

    with col_r2:
        st.markdown("#### Probability Breakdown")
        st.metric("🟢 Legitimate", f"{legit_prob * 100:.2f}%")
        st.progress(float(legit_prob))
        st.metric("🔴 Fraud", f"{fraud_prob * 100:.2f}%")
        st.progress(float(fraud_prob))

    # ── Risk level ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Risk Assessment")
    if fraud_prob < 0.2:
        st.success(f"🟢 **LOW RISK** — {fraud_prob*100:.1f}% fraud probability. Transaction appears safe.")
    elif fraud_prob < 0.5:
        st.warning(f"🟡 **MEDIUM RISK** — {fraud_prob*100:.1f}% fraud probability. Consider manual review.")
    else:
        st.error(f"🔴 **HIGH RISK** — {fraud_prob*100:.1f}% fraud probability. Transaction flagged!")

    # ── Input summary expander ─────────────────────────────────────────────
    with st.expander("🔎 View Full Input Feature Summary"):
        import pandas as pd
        df_in = pd.DataFrame({
            "Feature":      FEATURE_ORDER,
            "Raw Value":    [f"{raw_input[0][j]:.4f}"  for j in range(len(FEATURE_ORDER))],
            "Scaled Value": [f"{raw_scaled[0][j]:.4f}" for j in range(len(FEATURE_ORDER))],
        })
        st.dataframe(df_in, use_container_width=True, height=350)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#546e7a; font-size:0.85rem; padding:1rem 0;">
    💳 Credit Card Fraud Detection &nbsp;|&nbsp;
    ANN inference powered by <strong>NumPy + h5py</strong> &nbsp;|&nbsp;
    Dataset: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" 
               target="_blank" style="color:#42a5f5;">Kaggle ULB</a>
</div>
""", unsafe_allow_html=True)