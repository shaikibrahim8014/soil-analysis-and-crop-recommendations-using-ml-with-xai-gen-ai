import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
import plotly.express as px
import textwrap

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.prediction import predict_top3, model_available
from src.soil_health import analyze_soil
from src.explainability import explain_prediction, global_feature_importance
from src.database import initialize_database, save_prediction, get_prediction_history
from src.data_loader import load_dataset, validate_dataset_columns

# Gemini Advisor
from src.gemini_advisor import get_agriculture_advice

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Soil AI | Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATABASE
# ============================================================

initialize_database()


# ============================================================
# PREMIUM PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

 /* ============================================================
   SOIL CROP AI
   PROFESSIONAL AGRICULTURE + AI DASHBOARD THEME
   ============================================================ */


/* ============================================================
   GLOBAL
   ============================================================ */

html,
body,
[class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif !important;
}

.stApp {
    background: #f4f9f6 !important;
    color: #17211b !important;
}

.block-container {
    max-width: 1450px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}


/* ============================================================
   MAIN PAGE TITLE
   ============================================================ */

.main-title {
    margin-top: 18px;
    margin-bottom: 8px;

    color: #14532d !important;

    font-size: 2.45rem;
    font-weight: 850;

    letter-spacing: -1px;
    line-height: 1.2;
}

.subtitle {
    color: #64748b !important;

    font-size: 1rem;
    font-weight: 500;

    line-height: 1.6;
    margin-bottom: 25px;
}


/* ============================================================
   PROJECT BRAND
   ============================================================ */

.project-title {
    margin-top: 10px;
    margin-bottom: 3px;

    color: #14532d !important;

    font-size: 2.25rem;
    font-weight: 850;

    letter-spacing: -1px;
}

.project-subtitle {
    color: #64748b !important;

    font-size: 0.95rem;
    font-weight: 500;

    margin-bottom: 18px;
}


/* ============================================================
   HEADINGS
   ============================================================ */

h1,
h2,
h3,
h4,
h5,
h6 {
    color: #17211b !important;
    font-weight: 800 !important;
}

h3 {
    color: #166534 !important;
}

p,
li {
    color: #26332b !important;
    line-height: 1.75;
}

[data-testid="stMarkdownContainer"] p {
    color: #26332b !important;
}


/* ============================================================
   TOP NAVIGATION
   ============================================================ */

div[data-testid="stRadio"] {
    width: 100% !important;

    margin-top: 5px !important;
    margin-bottom: 20px !important;
}

div[data-testid="stRadio"] > div {
    width: 100% !important;

    display: flex !important;
    flex-direction: row !important;

    justify-content: center !important;
    align-items: center !important;

    flex-wrap: wrap !important;

    gap: 5px !important;

    padding: 7px !important;

    background: #ffffff !important;

    border: 1px solid #d8e8dd !important;

    border-radius: 16px !important;

    box-shadow:
        0 8px 25px rgba(22, 101, 52, 0.07) !important;
}

div[data-testid="stRadio"] label {
    margin: 0 !important;

    padding: 11px 17px !important;

    border-radius: 11px !important;

    background: transparent !important;

    color: #334155 !important;

    font-size: 0.84rem !important;
    font-weight: 650 !important;

    white-space: nowrap !important;

    cursor: pointer !important;

    transition: all 0.2s ease !important;
}

div[data-testid="stRadio"] label:hover {
    background: #ecfdf5 !important;
    color: #15803d !important;
}

div[data-testid="stRadio"] label:has(input:checked) {
    background: #166534 !important;

    color: #ffffff !important;

    box-shadow:
        0 5px 16px rgba(22, 101, 52, 0.22) !important;
}

div[data-testid="stRadio"] label:has(input:checked) * {
    color: #ffffff !important;
}

div[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {
    min-height: 115px !important;

    padding: 20px !important;

    background: #ffffff !important;

    border: 1px solid #dbe8df !important;

    border-radius: 18px !important;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.055) !important;

    transition: all 0.25s ease !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);

    box-shadow:
        0 15px 35px rgba(22, 101, 52, 0.12) !important;
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;

    font-size: 0.75rem !important;

    font-weight: 700 !important;

    text-transform: uppercase;

    letter-spacing: 0.5px;
}

[data-testid="stMetricValue"] {
    color: #166534 !important;

    font-size: 1.55rem !important;

    font-weight: 850 !important;
}


/* ============================================================
   SOIL INPUT SECTION
   ============================================================ */

.stNumberInput {
    background: #ffffff !important;

    border-radius: 14px !important;
}

.stNumberInput input {
    min-height: 45px !important;

    background: #ffffff !important;

    color: #17211b !important;

    border: 1px solid #cfe1d5 !important;

    border-radius: 11px !important;

    font-size: 0.94rem !important;

    font-weight: 500 !important;
}

.stNumberInput input:hover {
    border-color: #4ade80 !important;
}

.stNumberInput input:focus {
    border-color: #16a34a !important;

    box-shadow:
        0 0 0 3px rgba(22, 163, 74, 0.12) !important;
}

.stNumberInput label {
    color: #1f2937 !important;

    font-weight: 650 !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    width: 100% !important;

    min-height: 48px !important;

    border: none !important;

    border-radius: 12px !important;

    background: #166534 !important;

    color: #ffffff !important;

    font-size: 0.94rem !important;

    font-weight: 750 !important;

    box-shadow:
        0 7px 18px rgba(22, 101, 52, 0.20) !important;

    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    background: #15803d !important;

    transform: translateY(-2px) !important;

    box-shadow:
        0 12px 25px rgba(22, 101, 52, 0.25) !important;
}

.stButton > button * {
    color: #ffffff !important;
}


/* ============================================================
   HOME PAGE
   ============================================================ */

.home-page {
    background: #edf8f1 !important;

    padding: 30px;

    margin: -20px -20px 0 -20px;

    border-radius: 0 0 25px 25px;
}


/* ============================================================
   PROJECT OVERVIEW
   ============================================================ */

.intro-card {
    width: 100%;

    box-sizing: border-box;

    padding: 32px;

    margin-top: 22px;

    background: #ffffff !important;

    color: #26332b !important;

    border: 1px solid #d5e8db !important;

    border-radius: 22px !important;

    box-shadow:
        0 10px 30px rgba(22, 101, 52, 0.08) !important;
}

.intro-card h2 {
    margin-top: 0 !important;

    margin-bottom: 16px !important;

    color: #14532d !important;

    font-size: 1.65rem !important;

    font-weight: 850 !important;
}

.intro-card h3 {
    margin-top: 24px !important;

    margin-bottom: 10px !important;

    color: #166534 !important;

    font-size: 1.15rem !important;

    font-weight: 750 !important;
}

.intro-card p {
    margin-bottom: 15px !important;

    color: #374151 !important;

    font-size: 0.98rem !important;

    line-height: 1.8 !important;
}


/* ============================================================
   PROJECT WORKFLOW
   ============================================================ */

.workflow {
    margin-top: 20px;

    padding: 20px 25px;

    background: linear-gradient(
        135deg,
        #14532d,
        #166534
    ) !important;

    color: #ffffff !important;

    border-radius: 16px;

    text-align: center;

    font-size: 0.92rem;

    font-weight: 650;

    line-height: 1.9;

    box-shadow:
        0 8px 22px rgba(22, 101, 52, 0.18);
}

.workflow * {
    color: #ffffff !important;
}


/* ============================================================
   SOIL HEALTH CARD
   ============================================================ */

.result-card {
    padding: 24px 27px;

    margin: 16px 0;

    background: #ffffff !important;

    border: 1px solid #d8e8dd !important;

    border-left: 5px solid #16a34a !important;

    border-radius: 16px;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.055);
}

.result-card p {
    color: #26332b !important;
}


/* ============================================================
   CROP RECOMMENDATION CARDS
   ============================================================ */

.crop-card {
    padding: 25px;

    text-align: center;

    background: #ffffff !important;

    border: 1px solid #d8e8dd !important;

    border-radius: 18px;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.055);

    transition: all 0.25s ease;
}

.crop-card:hover {
    transform: translateY(-5px);

    border-color: #86efac !important;

    box-shadow:
        0 15px 35px rgba(22, 101, 52, 0.12);
}

.crop-name {
    color: #14532d !important;

    font-size: 1.1rem;

    font-weight: 850;
}

.crop-probability {
    margin-top: 9px;

    color: #16a34a !important;

    font-size: 1.55rem;

    font-weight: 850;
}

.small {
    margin-top: 7px;

    color: #64748b !important;

    font-size: 0.78rem;
}


/* ============================================================
   GEMINI AGRICULTURAL ADVISOR
   ============================================================ */

.ai-card {
    padding: 28px;

    margin-top: 16px;

    background: #f0f9ff !important;

    border: 1px solid #bae6fd !important;

    border-left: 5px solid #0284c7 !important;

    border-radius: 18px;

    box-shadow:
        0 8px 27px rgba(2, 132, 199, 0.08);
}

.ai-title {
    color: #075985 !important;

    font-size: 1.2rem;

    font-weight: 850;

    margin-bottom: 12px;
}

.ai-card p {
    color: #1f2937 !important;

    line-height: 1.8;
}


/* ============================================================
   XAI FEATURE CARD
   ============================================================ */

.xai-table {
    width: 100%;

    border-collapse: separate;

    border-spacing: 0;

    background: #ffffff !important;

    border: 1px solid #dbeafe;

    border-radius: 15px;

    overflow: hidden;

    box-shadow:
        0 8px 25px rgba(30, 64, 175, 0.06);

    margin-top: 12px;

    margin-bottom: 25px;
}

.xai-table th {
    padding: 14px 15px;

    background: #1e40af !important;

    color: #ffffff !important;

    font-size: 0.84rem;

    font-weight: 750;

    text-align: left;
}

.xai-table td {
    padding: 13px 15px;

    background: #ffffff !important;

    color: #1f2937 !important;

    font-size: 0.88rem;

    border-bottom: 1px solid #e5e7eb;
}

.xai-table tr:nth-child(even) td {
    background: #f8fbff !important;
}

.xai-table tbody tr:hover td {
    background: #eff6ff !important;

    color: #1e40af !important;
}


/* ============================================================
   GENERAL TABLES
   ============================================================ */

.premium-table,
.history-table {
    width: 100%;

    border-collapse: separate;

    border-spacing: 0;

    background: #ffffff !important;

    border: 1px solid #d8e8dd;

    border-radius: 15px;

    overflow: hidden;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.055);

    margin-top: 10px;

    margin-bottom: 25px;
}

.premium-table th,
.history-table th {
    padding: 14px 15px;

    background: #166534 !important;

    color: #ffffff !important;

    font-size: 0.84rem;

    font-weight: 750;

    text-align: left;
}

.premium-table td,
.history-table td {
    padding: 13px 15px;

    background: #ffffff !important;

    color: #1f2937 !important;

    font-size: 0.88rem;

    border-bottom: 1px solid #e5ebe7;
}

.premium-table tr:nth-child(even) td,
.history-table tr:nth-child(even) td {
    background: #f7fbf8 !important;
}

.premium-table tbody tr:hover td,
.history-table tbody tr:hover td {
    background: #ecfdf5 !important;

    color: #166534 !important;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {
    background: #ffffff !important;

    border: 1px solid #d8e8dd !important;

    border-radius: 15px !important;

    overflow: hidden !important;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.055) !important;

    color-scheme: light !important;
}


/* ============================================================
   MODEL PERFORMANCE / CONFUSION MATRIX
   ============================================================ */

.matrix-card {
    padding: 18px;

    background: #ffffff !important;

    border: 1px solid #dbe8df;

    border-radius: 18px;

    box-shadow:
        0 8px 27px rgba(15, 23, 42, 0.06);
}

[data-testid="stPlotlyChart"] {
    background: #ffffff !important;

    border-radius: 15px !important;
}


/* ============================================================
   ALERTS
   ============================================================ */

[data-testid="stAlert"] {
    border-radius: 13px !important;

    border: 1px solid #d5e6dc !important;
}

[data-testid="stAlert"] p {
    color: #1f2937 !important;
}


/* ============================================================
   EXPANDER
   ============================================================ */

[data-testid="stExpander"] {
    background: #ffffff !important;

    border: 1px solid #d8e8dd !important;

    border-radius: 14px !important;
}


/* ============================================================
   DIVIDER
   ============================================================ */

hr {
    border: none !important;

    height: 1px !important;

    background: #d8e8dd !important;

    margin: 22px 0 !important;
}


/* ============================================================
   CAPTION
   ============================================================ */

.stCaption {
    color: #64748b !important;
}


/* ============================================================
   REMOVE BLACK BACKGROUNDS
   ============================================================ */

.stCodeBlock,
.stCodeBlock pre,
pre,
code {
    background: #ffffff !important;

    color: #14532d !important;

    border-radius: 12px !important;
}


/* ============================================================
   MOBILE RESPONSIVE
   ============================================================ */

@media (max-width: 900px) {

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .project-title {
        font-size: 1.85rem;
    }

    .main-title {
        font-size: 2rem;
    }

    div[data-testid="stRadio"] > div {
        justify-content: flex-start !important;

        flex-wrap: nowrap !important;

        overflow-x: auto !important;
    }

    div[data-testid="stRadio"] label {
        padding: 9px 12px !important;

        font-size: 0.78rem !important;
    }

    .intro-card {
        padding: 20px !important;
    }

    .intro-card h2 {
        font-size: 1.35rem !important;
    }

    .intro-card h3 {
        font-size: 1.05rem !important;
    }

}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP NAVIGATION
# ============================================================

page = st.radio(
    "Navigation",
    [
        "Home",
        "Soil Analysis",
        "Crop Recommendation",
        "XAI Explanation",
        "Model Performance",
        "Prediction History",
        "About",
    ],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# NAVIGATION DIVIDER
# ============================================================

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# INPUT FORM
# ============================================================


def input_form():

    st.subheader("🌱 Soil Parameters")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        n = st.number_input("Nitrogen (kg/ha)", min_value=0.0, value=50.0)

    with c2:
        p = st.number_input("Phosphorus (kg/ha)", min_value=0.0, value=50.0)

    with c3:
        k = st.number_input("Potassium (kg/ha)", min_value=0.0, value=50.0)

    with c4:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)

    st.subheader("🌤️ Environmental Parameters")

    c1, c2, c3 = st.columns(3)

    with c1:
        temperature = st.number_input("Temperature (°C)", value=25.0)

    with c2:
        humidity = st.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0, value=70.0
        )

    with c3:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)

    return {
        "N": n,
        "P": p,
        "K": k,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="main-title">
            🌱 AI-Based Soil Analysis & Intelligent Crop Recommendation
        </div>

        <div class="subtitle">
            Machine Learning + Explainable AI + Gemini AI
            for Data-Driven Agriculture
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric("Machine Learning", "Classification")

    c2.metric("Explainable AI", "SHAP")

    c3.metric("AI Advisor", "Gemini")

    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="intro-card">

            <h2>🌱 Project Overview</h2>

            <p>
                This application accepts soil and environmental conditions,
                predicts suitable crops, ranks the Top-3 model
                recommendations, explains the prediction using SHAP,
                analyzes soil health, and provides agricultural guidance
                using Gemini AI.
            </p>

            <h3>🔄 Core Workflow</h3>

            <div class="workflow">
                User Input → Validation → ML Prediction →
                Top-3 Ranking → SHAP Explanation →
                Soil Health → Gemini Agricultural Advice →
                SQLite History
            </div>

            <h3>🎯 Project Objective</h3>

            <p>
                The main objective of this project is to provide an
                intelligent decision-support platform that combines
                Machine Learning, Explainable AI and Generative AI
                to support data-driven agricultural decisions.
            </p>

            <h3>⚠️ Important</h3>

            <p>
                Model probabilities are model-estimated probabilities
                or confidence values, not guarantees of farming success.
                Actual agricultural decisions should consider laboratory
                soil tests, local climate, agronomist guidance and
                crop-specific requirements.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SOIL ANALYSIS
# ============================================================

elif page == "Soil Analysis":

    st.markdown(
        """
        <div class="main-title">
            🔬 Soil Analysis
        </div>

        <div class="subtitle">
            Analyze soil health using nutrient and environmental parameters.
        </div>
        """,
        unsafe_allow_html=True,
    )

    values = input_form()

    if st.button("🔍 Analyze Soil", type="primary"):

        try:

            result = analyze_soil(**values)

            st.subheader("🌱 Soil Health Summary")

            cols = st.columns(4)

            for col, key in zip(cols, ["N", "P", "K", "ph"]):

                col.metric(key, result["status"][key])

            st.metric("Soil Health Score", f'{result["score"]:.1f}/100')

            st.info(result["disclaimer"])

        except ValueError as e:

            st.error(str(e))

        except Exception as e:

            st.error(f"Soil analysis failed: {e}")


# ============================================================
# CROP RECOMMENDATION
# ============================================================

elif page == "Crop Recommendation":

    st.markdown(
        """
        <div class="main-title">
            🌾 Crop Recommendation
        </div>

        <div class="subtitle">
            Predict and rank the most suitable crops using Machine Learning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_available():

        st.error("Trained model not found. Run: python src/train_model.py")

    values = input_form()

    if st.button("🌾 Recommend Crops", type="primary"):

        try:

            result = predict_top3(**values)

            predicted_crop = result["predicted_crop"]

            st.success(f"🌱 Predicted Crop: {predicted_crop}")

            # ------------------------------------------------
            # TOP 3
            # ------------------------------------------------

            st.subheader("🏆 Top 3 Recommended Crops")

            cols = st.columns(3)

            medals = ["🥇", "🥈", "🥉"]

            for col, medal, item in zip(cols, medals, result["top3"]):

                with col:

                    st.markdown(
                        f"""
                        <div class="crop-card">

                            <div class="crop-name">
                                {medal} {item["crop"]}
                            </div>

                            <div class="crop-probability">
                                {item["probability"]:.2%}
                            </div>

                            <div class="small">
                                Model-estimated probability
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.caption(
                "These are model-estimated probabilities, "
                "not crop-success probabilities."
            )

            # ------------------------------------------------
            # SOIL HEALTH
            # ------------------------------------------------

            health = analyze_soil(**values)

            st.subheader("🌱 Soil Health")

            st.metric("Score", f'{health["score"]:.1f}/100')

            # ------------------------------------------------
            # XAI
            # ------------------------------------------------

            explanation = explain_prediction(values)

            st.subheader("🧠 Why did the model recommend this crop?")

            st.markdown(
                f"""
                <div class="result-card">

                    <p>
                        {explanation["plain_english"]}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # GEMINI
            # ------------------------------------------------

            st.subheader("🤖 Gemini Agricultural Advisor")

            soil_data = {
                "Nitrogen (N)": values["N"],
                "Phosphorus (P)": values["P"],
                "Potassium (K)": values["K"],
                "Soil pH": values["ph"],
                "Temperature (°C)": values["temperature"],
                "Humidity (%)": values["humidity"],
                "Rainfall (mm)": values["rainfall"],
            }

            with st.spinner("🤖 Gemini is generating agricultural advice..."):

                advice = get_agriculture_advice(
                    soil_data=soil_data,
                    predicted_crop=predicted_crop,
                    explanation=explanation["plain_english"],
                )

            st.markdown(
                f"""
                <div class="ai-card">

                    <div class="ai-title">
                        🌾 AI Agricultural Advice
                    </div>

                    <p>
                        {advice}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            save_prediction(values, predicted_crop, result["top3"])

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# ============================================================
# XAI EXPLANATION
# ============================================================

elif page == "XAI Explanation":

    st.markdown(
        """
        <div class="main-title">
            🧠 Explainable AI
        </div>

        <div class="subtitle">
            Understand which features influenced the model prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_available():

        st.warning("Train the model first.")

    else:

        st.subheader("📊 Global Feature Importance")

        try:

            imp = global_feature_importance()

            st.bar_chart(imp.set_index("feature"))

        except Exception as e:

            st.error(f"Could not load global SHAP importance: {e}")

        st.subheader("🔍 Explain a New Prediction")

        values = input_form()

        if st.button("🧠 Explain Prediction", type="primary"):

            try:

                exp = explain_prediction(values)

                st.success(f'Predicted crop: {exp["predicted_crop"]}')

                contribution_df = exp["contributions"]

                st.markdown(
                    """
                    <h3>
                        📋 Feature Contributions
                    </h3>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    contribution_df.to_html(
                        index=False, classes="xai-table", escape=True
                    ),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="result-card">

                        <h3>💡 Explanation</h3>

                        <p>
                            {exp["plain_english"]}
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:

                st.error(f"XAI failed: {e}")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        """
        <div class="main-title">
            📊 Model Performance
        </div>

        <div class="subtitle">
            Compare trained models and inspect the confusion matrix.
        </div>
        """,
        unsafe_allow_html=True,
    )

    meta_path = ROOT / "models" / "model_metadata.json"

    if not meta_path.exists():

        st.warning("No model metadata found. Run the training script first.")

    else:

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # ------------------------------------------------
        # MODEL INFORMATION
        # ------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Selected Model", meta.get("selected_model", "N/A"))

        c2.metric("Classes", meta.get("num_classes", "N/A"))

        c3.metric("Training Samples", meta.get("training_samples", "N/A"))

        c4.metric("Testing Samples", meta.get("testing_samples", "N/A"))

        # ------------------------------------------------
        # MODEL COMPARISON
        # ------------------------------------------------

        if meta.get("model_comparison"):

            comparison_df = pd.DataFrame(meta["model_comparison"])

            st.markdown("<h3>📊 Model Comparison</h3>", unsafe_allow_html=True)

            st.markdown(
                comparison_df.to_html(
                    index=False, classes="premium-table", escape=True
                ),
                unsafe_allow_html=True,
            )

        # ------------------------------------------------
        # CONFUSION MATRIX
        # ------------------------------------------------

        if meta.get("confusion_matrix"):

            st.markdown("<h3>🔥 Confusion Matrix</h3>", unsafe_allow_html=True)

            st.markdown('<div class="matrix-card">', unsafe_allow_html=True)

            fig = px.imshow(
                meta["confusion_matrix"],
                text_auto=True,
                aspect="auto",
                color_continuous_scale=[
                    "#f0fdf4",
                    "#bbf7d0",
                    "#6ee7b7",
                    "#34d399",
                    "#059669",
                    "#064e3b",
                ],
            )

            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font={"color": "#111827", "family": "Segoe UI"},
                margin={"l": 30, "r": 30, "t": 25, "b": 30},
            )

            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )

            st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    st.markdown(
        """
        <div class="main-title">
            🗃️ Prediction History
        </div>

        <div class="subtitle">
            View previously generated crop recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = get_prediction_history()

    if history.empty:

        st.info("No predictions have been stored yet.")

    else:

        st.markdown("<h3>📋 Previous Predictions</h3>", unsafe_allow_html=True)

        st.markdown(
            history.to_html(index=False, classes="history-table", escape=True),
            unsafe_allow_html=True,
        )

# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        """
<div class="main-title">
    ℹ️ About Soil Crop AI
</div>

<div class="subtitle">
    Technology and architecture behind the intelligent agriculture platform.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="intro-card">

    <h2>
        ###🌱 AI-Based Soil Analysis and Intelligent Crop Recommendation
    </h2>

    <p>
        AI-Based Soil Analysis and Intelligent Crop Recommendation
        is an intelligent agriculture decision-support system
        that combines Machine Learning, Explainable AI,
        Soil Health Analysis and Generative AI.
    </p>

    <h3>🛠️ Technology Stack</h3>

    <p>
        <b>Python 3.12</b><br>
        <b>Pandas / NumPy</b><br>
        <b>Scikit-learn</b><br>
        <b>XGBoost</b><br>
        <b>SHAP</b><br>
        <b>Google Gemini AI</b><br>
        <b>Streamlit</b><br>
        <b>Plotly / Matplotlib</b><br>
        <b>Joblib</b><br>
        <b>SQLite</b>
    </p>

    <h3>🤖 Gemini AI</h3>

    <p>
        Gemini AI acts as an agricultural advisor.
        It receives soil N, P, K, pH, temperature,
        humidity, rainfall, the predicted crop and
        the XAI explanation.
    </p>

    <p>
        It then generates understandable agricultural guidance.
    </p>

</div>
""",
        unsafe_allow_html=True,
    )

    st.warning(
        "Disclaimer: This is a decision-support system. "
        "It is not a replacement for professional agricultural "
        "advice or laboratory soil testing."
    )
