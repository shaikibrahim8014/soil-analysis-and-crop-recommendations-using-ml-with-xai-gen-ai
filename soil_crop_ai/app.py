import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

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
from src.explainability import (
    explain_prediction,
    global_feature_importance,
)
from src.database import (
    initialize_database,
    save_prediction,
    get_prediction_history,
)

# Gemini is optional. The rest of the app should still open
# if Gemini dependencies or API configuration are unavailable.
try:
    from src.gemini_advisor import get_agriculture_advice

    GEMINI_AVAILABLE = True
    GEMINI_ERROR = None
except Exception as exc:
    get_agriculture_advice = None
    GEMINI_AVAILABLE = False
    GEMINI_ERROR = str(exc)

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Soil Crop AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

initialize_database()

# ============================================================
# PREMIUM FOREST-GREEN CSS
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --forest: #153f2b;
        --green: #1f6b45;
        --leaf: #3b8b5b;
        --mint: #eaf5ed;
        --cream: #fbfaf6;
        --text: #1d2c22;
        --muted: #66766b;
        --border: #dce8df;
        --shadow: 0 12px 30px rgba(21, 63, 43, .075);
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(
                900px 420px at 0% 0%,
                rgba(217, 240, 222, .62),
                transparent 68%
            ),
            linear-gradient(180deg, #f8fbf8, var(--cream)) !important;
        color: var(--text) !important;
    }

    .block-container {
        max-width: 1360px !important;
        padding: 1.6rem clamp(1rem, 3vw, 2.4rem) 3.5rem !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
        font-weight: 800 !important;
        letter-spacing: -.025em !important;
    }

    .main-title {
        margin: 15px 0 8px;
        color: var(--forest) !important;
        font-size: clamp(1.8rem, 3.4vw, 2.65rem);
        font-weight: 850;
        letter-spacing: -.045em;
        line-height: 1.2;
    }

    .subtitle {
        color: var(--muted) !important;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 22px;
    }

    p, li {
        color: #34463a;
        line-height: 1.75;
    }

    /* Navigation */

    div[data-testid="stRadio"] {
        width: 100% !important;
        margin: 8px 0 20px !important;
    }

    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 6px !important;
        padding: 8px !important;
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        box-shadow: 0 7px 24px rgba(21, 63, 43, .055) !important;
    }

    div[data-testid="stRadio"] label {
        margin: 0 !important;
        padding: 10px 14px !important;
        border: 1px solid transparent !important;
        border-radius: 12px !important;
        color: #536458 !important;
        font-size: .84rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        transition: background .18s ease, color .18s ease;
    }

    div[data-testid="stRadio"] label:hover {
        background: var(--mint) !important;
        color: var(--green) !important;
    }

    div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(
            135deg, var(--forest), var(--green)
        ) !important;
        color: #ffffff !important;
        box-shadow: 0 5px 15px rgba(21, 63, 43, .2) !important;
    }

    div[data-testid="stRadio"] label:has(input:checked) * {
        color: #ffffff !important;
    }

    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* Metric cards */

    [data-testid="stMetric"] {
        min-height: 108px !important;
        padding: 20px !important;
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        box-shadow: var(--shadow) !important;
        transition: transform .2s ease, box-shadow .2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 17px 34px rgba(21, 63, 43, .11) !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: .75rem !important;
        font-weight: 750 !important;
        letter-spacing: .06em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--forest) !important;
        font-weight: 850 !important;
    }

    /* Input fields */

    .stNumberInput input,
    .stTextInput input,
    .stTextArea textarea,
    [data-baseweb="select"] > div {
        min-height: 44px !important;
        border: 1px solid #d2e1d6 !important;
        border-radius: 11px !important;
        background: #ffffff !important;
        color: var(--text) !important;
    }

    .stNumberInput label,
    .stTextInput label,
    .stSelectbox label {
        color: #304638 !important;
        font-weight: 680 !important;
    }

    /* Buttons */

    .stButton > button,
    .stDownloadButton > button {
        width: 100% !important;
        min-height: 46px !important;
        border: 1px solid var(--green) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        background: linear-gradient(
            135deg, var(--forest), var(--green)
        ) !important;
        font-weight: 750 !important;
        box-shadow: 0 7px 17px rgba(21, 63, 43, .16) !important;
        transition: transform .18s ease, box-shadow .18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(21, 63, 43, .22) !important;
    }

    .stButton > button *,
    .stDownloadButton > button * {
        color: #ffffff !important;
    }

    /* Dashboard cards */

    .intro-card, .crop-card, .result-card, .ai-card {
        box-sizing: border-box;
        border-radius: 20px;
        box-shadow: var(--shadow);
    }

    .intro-card {
        padding: 28px;
        margin: 12px 0 20px;
        background: #ffffff;
        border: 1px solid var(--border);
    }

    .intro-card h2 {
        margin-top: 0;
        color: var(--forest) !important;
    }

    .crop-card {
        min-height: 155px;
        padding: 24px 16px;
        text-align: center;
        background: linear-gradient(180deg, #ffffff, #f5faf6);
        border: 1px solid var(--border);
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .crop-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 35px rgba(21, 63, 43, .12);
    }

    .feature-icon {
        font-size: 36px;
        margin-bottom: 12px;
    }

    .crop-name {
        color: var(--forest) !important;
        font-size: 1rem;
        font-weight: 850;
        overflow-wrap: anywhere;
    }

    .crop-probability {
        color: var(--green) !important;
        font-size: 1.5rem;
        font-weight: 850;
    }

    .small {
        color: var(--muted) !important;
        font-size: .82rem;
        margin-top: 7px;
    }

    .result-card {
        padding: 22px 26px;
        margin: 16px 0;
        background: #ffffff;
        border: 1px solid var(--border);
        border-left: 4px solid var(--leaf);
    }

    .ai-card {
        padding: 25px;
        margin: 16px 0;
        background: #f0f8f2;
        border: 1px solid #cfe3d5;
        border-left: 4px solid var(--leaf);
    }

    .ai-title {
        color: var(--forest) !important;
        font-size: 1.15rem;
        font-weight: 850;
        margin-bottom: 10px;
    }

    /* Tables and charts */

    .xai-table, .premium-table, .history-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 15px;
        background: #ffffff;
        box-shadow: 0 6px 20px rgba(21, 63, 43, .045);
        margin: 12px 0 24px;
    }

    .xai-table th, .premium-table th, .history-table th {
        padding: 13px 15px;
        text-align: left;
        color: #ffffff !important;
        background: var(--forest) !important;
        font-size: .84rem;
        font-weight: 750;
    }

    .xai-table td, .premium-table td, .history-table td {
        padding: 12px 15px;
        color: #2b3b30 !important;
        background: #ffffff !important;
        border-bottom: 1px solid #e8eee9;
        font-size: .88rem;
    }

    .xai-table tr:nth-child(even) td,
    .premium-table tr:nth-child(even) td,
    .history-table tr:nth-child(even) td {
        background: #f5faf6 !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"],
    [data-testid="stExpander"] {
        overflow: hidden !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        background: #ffffff !important;
        box-shadow: 0 6px 22px rgba(21, 63, 43, .045) !important;
    }

    [data-testid="stAlert"] {
        border-radius: 14px !important;
    }

    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(
            90deg, transparent, #d6e5da 12%,
            #d6e5da 88%, transparent
        ) !important;
        margin: 22px 0 !important;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 1rem .9rem 2.5rem !important;
        }

        div[data-testid="stRadio"] > div {
            flex-wrap: nowrap !important;
            justify-content: flex-start !important;
            overflow-x: auto !important;
        }

        div[data-testid="stRadio"] label {
            padding: 9px 12px !important;
            font-size: .78rem !important;
        }

        .intro-card {
            padding: 20px;
        }
    }

    @media (max-width: 560px) {
        .main-title {
            font-size: 1.75rem;
        }

        .subtitle {
            font-size: .92rem;
        }

        [data-testid="stMetric"] {
            padding: 16px !important;
        }

        .crop-card {
            padding: 18px 12px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# NAVIGATION
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

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# REUSABLE PAGE TITLE
# ============================================================


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="main-title">{html.escape(title)}</div>
        <div class="subtitle">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SOIL INPUT FORM
# ============================================================


def input_form():
    st.subheader("🌱 Soil Parameters")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        n = st.number_input("Nitrogen (N)", min_value=0.0, value=50.0, key="input_n")

    with c2:
        p = st.number_input("Phosphorus (P)", min_value=0.0, value=50.0, key="input_p")

    with c3:
        k = st.number_input("Potassium (K)", min_value=0.0, value=50.0, key="input_k")

    with c4:
        ph = st.number_input(
            "Soil pH", min_value=0.0, max_value=14.0, value=6.5, key="input_ph"
        )

    st.subheader("🌤️ Environmental Parameters")

    c1, c2, c3 = st.columns(3)

    with c1:
        temperature = st.number_input("Temperature (°C)", value=25.0, key="input_temp")

    with c2:
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            key="input_humidity",
        )

    with c3:
        rainfall = st.number_input(
            "Rainfall (mm)", min_value=0.0, value=100.0, key="input_rainfall"
        )

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
# HOME PAGE — SHORT DASHBOARD
# ============================================================

if page == "Home":

    page_header(
        "🌱 Soil Crop AI",
        "Intelligent Soil Analysis & Crop Recommendation",
    )

    st.markdown(
        """
        <div class="intro-card">
            <h2>Welcome to Soil Crop AI 🌿</h2>
            <p>
                Smart agriculture powered by Machine Learning,
                Explainable AI and Generative AI.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="crop-card">
                <div class="feature-icon">🔬</div>
                <div class="crop-name">Soil Analysis</div>
                <div class="small">
                    Analyze soil nutrients and health.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="crop-card">
                <div class="feature-icon">🌾</div>
                <div class="crop-name">Crop Recommendation</div>
                <div class="small">
                    Discover suitable crops using ML.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="crop-card">
                <div class="feature-icon">🤖</div>
                <div class="crop-name">AI Agriculture Advisor</div>
                <div class="small">
                    Get agricultural guidance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric("Technology", "Machine Learning")
    col2.metric("Explainability", "SHAP / XAI")
    col3.metric("AI Assistant", "Gemini AI")


# ============================================================
# SOIL ANALYSIS
# ============================================================

# ============================================================
# SOIL ANALYSIS
# ============================================================

if page == "Soil Analysis":

    page_header(
        "🌱 Soil Analysis",
        "Check soil health, nutrient balance, and environmental conditions.",
    )

    st.markdown("### Enter Soil Parameters")

    col1, col2 = st.columns(2)

    with col1:
        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=50.0,
            step=1.0,
            key="health_n",
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=40.0,
            step=1.0,
            key="health_p",
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=40.0,
            step=1.0,
            key="health_k",
        )

        soil_ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
            key="health_ph",
        )

    with col2:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=25.0,
            step=0.5,
            key="health_temp",
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
            key="health_humidity",
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=100.0,
            step=1.0,
            key="health_rainfall",
        )

    st.caption(
        "Enter measured soil values when available. "
        "The score is an indicator, not a laboratory test."
    )

    if st.button(
        "🔬 Analyze Soil",
        type="primary",
        key="run_soil_health",
    ):
        try:
            # Run the soil-health function only once per click.
            health = analyze_soil(
                N=nitrogen,
                P=phosphorus,
                K=potassium,
                temperature=temperature,
                humidity=humidity,
                ph=soil_ph,
                rainfall=rainfall,
            )

            if not isinstance(health, dict):
                st.error(
                    "The soil analysis function did not return "
                    "the expected result dictionary."
                )
            elif "score" not in health:
                st.error(
                    "The soil analysis result has no 'score' field. "
                    "Check src/soil_health.py."
                )
            else:
                score = max(0.0, min(100.0, float(health["score"])))

                # Overall health status
                if score >= 80:
                    status = "Excellent"
                    st.success("🌿 Soil Health Status: Excellent")
                elif score >= 65:
                    status = "Good"
                    st.success("✅ Soil Health Status: Good")
                elif score >= 50:
                    status = "Moderate"
                    st.warning("⚠️ Soil Health Status: Moderate")
                else:
                    status = "Needs Improvement"
                    st.error("🛠️ Soil Health Status: Needs Improvement")

                st.markdown("---")
                st.subheader("📊 Soil Health Results")

                # One clear score and status
                score_col, status_col = st.columns(2)

                with score_col:
                    st.metric(
                        "Soil Health Score",
                        f"{score:.1f}/100",
                    )

                with status_col:
                    st.metric(
                        "Overall Status",
                        status,
                    )

                st.progress(int(round(score)))

                # Display the input values once
                st.subheader("🧪 Soil Parameter Summary")

                parameter_data = {
                    "Parameter": [
                        "Nitrogen (N)",
                        "Phosphorus (P)",
                        "Potassium (K)",
                        "Soil pH",
                        "Temperature",
                        "Humidity",
                        "Rainfall",
                    ],
                    "Measured Value": [
                        f"{nitrogen:.1f}",
                        f"{phosphorus:.1f}",
                        f"{potassium:.1f}",
                        f"{soil_ph:.1f}",
                        f"{temperature:.1f} °C",
                        f"{humidity:.1f}%",
                        f"{rainfall:.1f} mm",
                    ],
                }

                st.dataframe(
                    pd.DataFrame(parameter_data),
                    use_container_width=True,
                    hide_index=True,
                )

                # Show additional details returned by the function.
                # Avoid displaying the score a second time.
                details = {
                    key: value for key, value in health.items() if key != "score"
                }

                if details:
                    st.subheader("📝 Analysis Details")

                    for key, value in details.items():
                        label = str(key).replace("_", " ").title()
                        st.write(f"**{label}:** {value}")

                st.caption(
                    "Status thresholds are display categories for "
                    "the returned score. They are not universal "
                    "agricultural soil-quality standards."
                )

        except Exception as exc:
            st.error(f"Soil analysis failed: {exc}")


# ============================================================
# CROP RECOMMENDATION
# ============================================================

# ============================================================
# CROP RECOMMENDATION
# ============================================================

elif page == "Crop Recommendation":

    page_header(
        "🌾 Crop Recommendation",
        "Discover suitable crops using soil and environmental data.",
    )

    if not model_available():
        st.error(
            "Trained model not found. Train the model using "
            "`python -m src.train_model`."
        )

    else:
        values = input_form()

        if st.button("🌾 Recommend Crops", type="primary"):

            try:
                # ------------------------------------------------
                # 1. Predict the top 3 crops
                # ------------------------------------------------
                result = predict_top3(**values)
                predicted_crop = result["predicted_crop"]
                top3 = result.get("top3", [])

                st.success(f"🌱 Best Recommended Crop: {predicted_crop}")

                # ------------------------------------------------
                # 2. Display top 3 crops in separate cards
                # ------------------------------------------------
                st.subheader("🏆 Top 3 Recommended Crops")

                if not top3:
                    st.warning("No crop recommendations were returned.")

                else:
                    cols = st.columns(3, gap="medium")
                    medals = ["🥇", "🥈", "🥉"]

                    for index, (col, item) in enumerate(zip(cols, top3[:3])):
                        crop_name = html.escape(str(item["crop"]))

                        # Convert probability to a percentage safely.
                        probability = float(item["probability"])

                        if probability > 1:
                            probability = probability / 100

                        probability = max(0.0, min(1.0, probability))

                        with col:
                            with st.container(border=True):
                                st.markdown(
                                    f"""
                                    <div style="
                                        text-align: center;
                                        padding: 12px 5px 8px 5px;
                                    ">
                                        <div style="font-size: 36px;">
                                            {medals[index]}
                                        </div>
                                        <div style="
                                            font-size: 19px;
                                            font-weight: 800;
                                            color: #153f2b;
                                            margin: 12px 0;
                                            overflow-wrap: anywhere;
                                        ">
                                            {crop_name}
                                        </div>
                                        <div style="
                                            font-size: 28px;
                                            font-weight: 850;
                                            color: #1f6b45;
                                        ">
                                            {probability:.1%}
                                        </div>
                                        <div style="
                                            font-size: 12px;
                                            color: #66766b;
                                            margin-top: 5px;
                                        ">
                                            Estimated probability
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                st.progress(probability)

                st.caption(
                    "Probabilities are model estimates, not guarantees "
                    "of crop yield or farming success."
                )

                # ------------------------------------------------
                # 3. Soil health score
                # ------------------------------------------------
                st.markdown("---")
                st.subheader("🌱 Soil Health Assessment")

                health = analyze_soil(**values)

                if isinstance(health, dict) and "score" in health:
                    score = max(
                        0.0,
                        min(100.0, float(health["score"])),
                    )

                    if score >= 80:
                        health_status = "Excellent"
                    elif score >= 65:
                        health_status = "Good"
                    elif score >= 50:
                        health_status = "Moderate"
                    else:
                        health_status = "Needs Improvement"

                    health_col1, health_col2 = st.columns(2)

                    with health_col1:
                        st.metric("Soil Health Score", f"{score:.1f}/100")

                    with health_col2:
                        st.metric("Soil Health Status", health_status)

                    st.progress(int(round(score)))

                    # Show other details without repeating the score.
                    for key, value in health.items():
                        if key != "score":
                            label = str(key).replace("_", " ").title()
                            st.write(f"**{label}:** {value}")

                else:
                    st.info(
                        "Crop recommendation succeeded, but the soil "
                        "health score was unavailable."
                    )

                # ------------------------------------------------
                # 4. Explainable AI
                # ------------------------------------------------
                st.markdown("---")
                st.subheader("🧠 Why Was This Crop Recommended?")

                try:
                    explanation = explain_prediction(values)

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <p>
                                {html.escape(
                                    str(explanation["plain_english"])
                                )}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as exc:
                    st.warning(f"Could not generate the explanation: {exc}")
                    explanation = None

                # ------------------------------------------------
                # 5. Gemini agricultural advice
                # ------------------------------------------------
                st.markdown("---")
                st.subheader("🤖 AI Agricultural Advisor")

                if not GEMINI_AVAILABLE:
                    st.warning(
                        "Gemini advisor is unavailable. Check your "
                        "dependencies and API key configuration."
                    )

                    if GEMINI_ERROR:
                        st.caption(GEMINI_ERROR)

                else:
                    soil_data = {
                        "Nitrogen (N)": values["N"],
                        "Phosphorus (P)": values["P"],
                        "Potassium (K)": values["K"],
                        "Soil pH": values["ph"],
                        "Temperature (°C)": values["temperature"],
                        "Humidity (%)": values["humidity"],
                        "Rainfall (mm)": values["rainfall"],
                    }

                    try:
                        with st.spinner("Generating agricultural advice..."):
                            advice = get_agriculture_advice(
                                soil_data=soil_data,
                                predicted_crop=predicted_crop,
                                explanation=(
                                    explanation["plain_english"]
                                    if explanation
                                    else "No explanation available."
                                ),
                            )

                        st.markdown(
                            f"""
                            <div class="ai-card">
                                <div class="ai-title">
                                    🌾 Agricultural Advice
                                </div>
                                <p>{html.escape(str(advice))}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    except Exception as exc:
                        st.warning(f"Gemini advice failed: {exc}")

                # ------------------------------------------------
                # 6. Save prediction history
                # ------------------------------------------------
                try:
                    save_prediction(values, predicted_crop, top3)
                    st.success("✅ Prediction saved to history.")

                except Exception as exc:
                    st.warning(f"Could not save prediction history: {exc}")

            except Exception as exc:
                st.error(f"Crop recommendation failed: {exc}")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    page_header(
        "📊 Model Performance",
        "Inspect model metadata and evaluation results.",
    )

    metadata_path = ROOT / "models" / "model_metadata.json"

    if not metadata_path.exists():
        st.warning("Model metadata was not found. " "Run the training script first.")

    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Selected Model",
                metadata.get("selected_model", "N/A"),
            )
            c2.metric(
                "Classes",
                metadata.get("num_classes", "N/A"),
            )
            c3.metric(
                "Training Samples",
                metadata.get("training_samples", "N/A"),
            )
            c4.metric(
                "Testing Samples",
                metadata.get("testing_samples", "N/A"),
            )

            comparison = metadata.get("model_comparison")

            if comparison:
                st.subheader("📊 Model Comparison")

                comparison_df = pd.DataFrame(comparison)

                st.markdown(
                    comparison_df.to_html(
                        index=False,
                        classes="premium-table",
                        escape=True,
                    ),
                    unsafe_allow_html=True,
                )

            matrix = metadata.get("confusion_matrix")

            if matrix:
                st.subheader("🔥 Confusion Matrix")

                fig = px.imshow(
                    matrix,
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
                    font={
                        "color": "#1d2c22",
                        "family": "Segoe UI",
                    },
                    margin={"l": 30, "r": 30, "t": 25, "b": 30},
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        except Exception as exc:
            st.error(f"Could not load model metadata: {exc}")


# ============================================================
# PREDICTION HISTORY
# ============================================================

# ============================================================
# PREDICTION HISTORY
# ============================================================

# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    page_header(
        "🗃️ Prediction History",
        "View previous predictions with their top 3 crops and probabilities.",
    )

    try:
        history = get_prediction_history()

        if history is None or history.empty:
            st.info(
                "No prediction history available. "
                "Generate a crop recommendation first."
            )

        else:
            # Summary
            st.subheader("📊 History Overview")

            total_predictions = len(history)

            if "predicted_crop" in history.columns:
                unique_crops = history["predicted_crop"].nunique()
                latest_crop = str(history["predicted_crop"].iloc[0])
            else:
                unique_crops = 0
                latest_crop = "Not available"

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predictions", total_predictions)
            col2.metric("Different Crops", unique_crops)
            col3.metric("Latest Recommended Crop", latest_crop)

            st.markdown("---")

            # CSV export
            st.download_button(
                "⬇️ Download Prediction History (CSV)",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="soil_crop_prediction_history.csv",
                mime="text/csv",
                key="history_csv_download",
            )

            # Build a display table.
            # This supports either separate top-3 columns or a saved
            # top-3 list/dictionary, depending on the database schema.
            display_history = history.copy()

            def format_probability(value):
                try:
                    probability = float(value)
                    if probability > 1:
                        probability /= 100.0
                    return f"{probability:.2%}"
                except (TypeError, ValueError):
                    return str(value) if value is not None else "N/A"

            def format_top3(row):
                items = []

                # Case 1: Top-3 crops were stored as a JSON string.
                for column in ("top3", "top_3", "recommendations"):
                    if column in row.index:
                        raw = row[column]

                        if isinstance(raw, str):
                            try:
                                raw = json.loads(raw)
                            except (json.JSONDecodeError, TypeError):
                                raw = None

                        if isinstance(raw, list):
                            for item in raw[:3]:
                                if isinstance(item, dict):
                                    crop = item.get("crop", "Unknown")
                                    prob = item.get(
                                        "probability",
                                        item.get("prob", None),
                                    )

                                    if prob is not None:
                                        items.append(
                                            f"{crop} ({format_probability(prob)})"
                                        )
                                    else:
                                        items.append(str(crop))

                                else:
                                    items.append(str(item))

                            if items:
                                return "\n".join(
                                    f"{i + 1}. {item}" for i, item in enumerate(items)
                                )

                # Case 2: Top-3 crops were saved in separate columns.
                for rank in range(1, 4):
                    crop_keys = (
                        f"top{rank}_crop",
                        f"crop_{rank}",
                        f"recommended_crop_{rank}",
                    )
                    prob_keys = (
                        f"top{rank}_probability",
                        f"probability_{rank}",
                        f"crop_{rank}_probability",
                    )

                    crop = next(
                        (
                            row[key]
                            for key in crop_keys
                            if key in row.index and pd.notna(row[key])
                        ),
                        None,
                    )

                    prob = next(
                        (
                            row[key]
                            for key in prob_keys
                            if key in row.index and pd.notna(row[key])
                        ),
                        None,
                    )

                    if crop is not None:
                        if prob is not None:
                            items.append(
                                f"{rank}. {crop} " f"({format_probability(prob)})"
                            )
                        else:
                            items.append(f"{rank}. {crop}")

                return "\n".join(items) if items else "Not stored"

            # Combine crop names and probabilities in readable columns.
            display_history["Top 3 Recommended Crops"] = history.apply(
                format_top3,
                axis=1,
            )

            # Keep the original prediction columns while avoiding
            # duplicate display of the raw Top-3 storage field.
            raw_top3_columns = [
                column
                for column in (
                    "top3",
                    "top_3",
                    "recommendations",
                )
                if column in display_history.columns
            ]

            display_history = display_history.drop(
                columns=raw_top3_columns,
                errors="ignore",
            )

            # Search
            st.subheader("🔍 Search Prediction History")

            search_text = st.text_input(
                "Search by crop or parameter",
                key="prediction_history_search",
            )

            if search_text.strip():
                mask = (
                    display_history.astype(str)
                    .apply(
                        lambda column: column.str.contains(
                            search_text.strip(),
                            case=False,
                            regex=False,
                            na=False,
                        )
                    )
                    .any(axis=1)
                )

                display_history = display_history[mask]

            st.caption(f"{len(display_history)} records found.")

            # Table
            st.subheader("📋 Prediction Records")

            st.dataframe(
                display_history,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Top 3 Recommended Crops": st.column_config.TextColumn(
                        "Top 3 Recommended Crops",
                        help="Each crop is shown with its probability.",
                        width="large",
                    ),
                },
            )

    except Exception as exc:
        st.error(f"Could not load prediction history: {exc}")


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    page_header(
        "ℹ️ About Soil Crop AI",
        "AI-powered tools for agricultural decision support.",
    )

    st.markdown(
        """
        <div class="intro-card">
            <h2>🌱 Soil Crop AI</h2>
            <p>
                An agricultural decision-support application combining
                Machine Learning, soil-health analysis, Explainable AI
                and Generative AI.
            </p>
            <h3>🛠️ Technology Stack</h3>
            <p>
                Python · Streamlit · Pandas · Scikit-learn ·
                SHAP · Plotly · SQLite · Google Gemini AI
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "This application is a decision-support tool. It does not "
        "replace laboratory soil testing or local agricultural advice."
    )
