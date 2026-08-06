# 1. Import Libraries
import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif;
    }

    :root {
        --teal-deep: #0B3D3A;
        --teal: #146B63;
        --teal-bright: #1FA898;
        --mint: #E6F5F2;
        --coral: #E4572E;
        --coral-soft: #FBE4DC;
        --ink: #0F1B1A;
        --paper: #F7FAF9;
    }

    .stApp { background: var(--paper); }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, var(--teal-deep) 0%, var(--teal) 55%, var(--teal-bright) 100%);
        padding: 2.6rem 2.4rem;
        border-radius: 20px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(11, 61, 58, 0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2.3rem;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: #D6F0EC;
        font-size: 1.02rem;
        margin: 0;
        max-width: 620px;
    }

    /* Section card */
    .panel {
        background: white;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        border: 1px solid #E4EFEC;
        box-shadow: 0 2px 10px rgba(15, 27, 26, 0.04);
        margin-bottom: 1.2rem;
    }
    .panel h3 {
        color: var(--teal-deep);
        font-size: 1.1rem;
        margin-top: 0;
    }

    /* Result cards */
    .result-card {
        border-radius: 18px;
        padding: 1.8rem 2rem;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
    }
    .result-high {
        background: linear-gradient(135deg, #FBE4DC, #FCEEE9);
        border: 1.5px solid #E4572E;
    }
    .result-low {
        background: linear-gradient(135deg, #E6F5F2, #F0FAF8);
        border: 1.5px solid #1FA898;
    }
    .result-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem; }
    .result-high .result-title { color: #B8391A; }
    .result-low .result-title { color: #0B3D3A; }
    .result-sub { color: #445; font-size: 0.95rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--teal-deep);
    }
    section[data-testid="stSidebar"] * { color: #E6F5F2 !important; }
    section[data-testid="stSidebar"] input {
        color: var(--ink) !important;
    }

    /* Buttons */
    div.stButton > button {
        background: var(--coral);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.15s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #C7481F;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(228, 87, 46, 0.3);
    }

    .metric-chip {
        display: inline-block;
        background: var(--mint);
        color: var(--teal-deep);
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING
# =========================================================
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "diabetes.csv")

@st.cache_data(show_spinner=False)
def load_data(path_or_buffer):
    data = pd.read_csv(path_or_buffer)
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        data[col] = data[col].replace(0, np.nan)
        data[col] = data[col].fillna(data[col].median())
    return data

@st.cache_resource(show_spinner=False)
def train_models(data: pd.DataFrame):
    """Train two candidate models so their accuracy can be compared side by side."""
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

    models = {
        "Random Forest": {"model": rf_model, "accuracy": rf_acc},
        "Logistic Regression": {"model": lr_model, "accuracy": lr_acc},
    }
    return models, scaler, list(X.columns)

# --- Sidebar: data source ---
with st.sidebar:
    st.markdown("### 📁 Dataset")
    uploaded = st.file_uploader("Upload diabetes.csv (optional)", type=["csv"])
    st.caption("If nothing is uploaded, the app looks for `diabetes.csv` next to this script.")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write(
        "This tool trains a Random Forest classifier on the Pima Indians "
        "Diabetes dataset and estimates diabetes risk from 8 health metrics."
    )
    st.markdown("---")
    st.caption("⚠️ Educational demo only — not a medical diagnosis.")

data_source = uploaded if uploaded is not None else (DEFAULT_PATH if os.path.exists(DEFAULT_PATH) else None)

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Risk Assessment</h1>
    <p>Enter a few health metrics and get an instant, model-driven estimate of diabetes risk —
    powered by a Random Forest trained on real clinical data.</p>
</div>
""", unsafe_allow_html=True)

if data_source is None:
    st.error(
        "No dataset found. Upload a `diabetes.csv` file from the sidebar, or place one "
        "named `diabetes.csv` in the same folder as this script.\n\n"
        "Expected columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, "
        "BMI, DiabetesPedigreeFunction, Age, Outcome."
    )
    st.stop()

with st.spinner("Loading data & training models..."):
    df = load_data(data_source)
    models, scaler, feature_names = train_models(df)

if "history" not in st.session_state:
    st.session_state.history = []  # each entry: dict with inputs, model, result, probability, time

# =========================================================
# MAIN LAYOUT
# =========================================================
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📝 Your Health Metrics")

    model_choice = st.selectbox(
        "Model", list(models.keys()),
        help="Compare predictions between two different classifiers trained on the same data.",
    )

    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        bp = st.number_input("Blood Pressure (mm Hg)", 0, 140, 70)
        insulin = st.number_input("Insulin (mu U/mL)", 0, 900, 85)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
    with c2:
        glucose = st.number_input("Glucose Level (mg/dL)", 0, 200, 120)
        skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        age = st.number_input("Age", 10, 100, 30)

    predict_clicked = st.button("🔍 Predict My Risk")
    st.markdown('</div>', unsafe_allow_html=True)

    active_model = models[model_choice]["model"]
    active_acc = models[model_choice]["accuracy"]

    chips = f'<span class="metric-chip">Model: {model_choice}</span>' \
            f'<span class="metric-chip">Test Accuracy: {active_acc*100:.1f}%</span>' \
            f'<span class="metric-chip">Training rows: {len(df)}</span>'
    st.markdown(chips, unsafe_allow_html=True)

    with st.expander("⚖️ Compare both models' accuracy"):
        comp_df = pd.DataFrame(
            [{"Model": name, "Test Accuracy (%)": round(info["accuracy"] * 100, 2)}
             for name, info in models.items()]
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📊 Result")

    if predict_clicked:
        input_df = pd.DataFrame(
            [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=feature_names,
        )
        input_scaled = scaler.transform(input_df)
        proba = active_model.predict_proba(input_scaled)[0]
        result = int(proba[1] >= 0.5)
        risk_pct = proba[1] * 100

        # Save to session history (keep only the most recent 5)
        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "model": model_choice,
            "risk_pct": risk_pct,
            "result": "Elevated Risk" if result == 1 else "Lower Risk",
            "inputs": input_df.iloc[0].to_dict(),
        })
        st.session_state.history = st.session_state.history[:5]

        if result == 1:
            st.markdown(f"""
            <div class="result-card result-high">
                <div class="result-title">🟥 Elevated Risk</div>
                <div class="result-sub">The model estimates a <b>{risk_pct:.1f}%</b> likelihood of diabetes based on the values entered.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-low">
                <div class="result-title">🟩 Lower Risk</div>
                <div class="result-sub">The model estimates a <b>{risk_pct:.1f}%</b> likelihood of diabetes based on the values entered.</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(min(max(proba[1], 0.0), 1.0))
        st.caption("Predicted probability of a diabetic outcome")

        with st.expander("See what you entered"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)

        st.info(
            "This is a statistical estimate from a machine-learning model, not a medical "
            "diagnosis. Please consult a healthcare professional for an accurate assessment.",
            icon="ℹ️",
        )
    else:
        st.write("Fill in your metrics on the left and click **Predict My Risk** to see your result here.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RECENT PREDICTIONS (session-only history, last 5)
# =========================================================
st.markdown('<div class="panel">', unsafe_allow_html=True)
hist_header, hist_clear = st.columns([4, 1])
with hist_header:
    st.markdown("### 🕒 Your Recent Predictions (this session)")
with hist_clear:
    if st.session_state.history and st.button("Clear", key="clear_history"):
        st.session_state.history = []
        st.rerun()

if not st.session_state.history:
    st.caption("No predictions yet this session. Run a prediction above to see it logged here.")
else:
    hist_df = pd.DataFrame([
        {
            "Time": h["time"],
            "Model": h["model"],
            "Result": h["result"],
            "Risk %": f'{h["risk_pct"]:.1f}%',
            "Glucose": h["inputs"]["Glucose"],
            "BMI": h["inputs"]["BMI"],
            "Age": h["inputs"]["Age"],
        }
        for h in st.session_state.history
    ])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
    st.caption(
        "Stored only in your browser session (not saved to a database) — it resets when you close or reload the app."
    )
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# DATA INSIGHTS (extra polish, optional expand)
# =========================================================
with st.expander("📈 Explore the training dataset"):
    tab1, tab2 = st.tabs(["Summary stats", "Outcome balance"])
    with tab1:
        st.dataframe(df.describe().T, use_container_width=True)
    with tab2:
        counts = df['Outcome'].value_counts().rename({0: "Non-diabetic", 1: "Diabetic"})
        st.bar_chart(counts)
