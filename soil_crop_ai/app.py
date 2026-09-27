import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction import predict_top3, model_available
from src.soil_health import analyze_soil
from src.explainability import explain_prediction, global_feature_importance
from src.database import initialize_database, save_prediction, get_prediction_history
from src.data_loader import load_dataset, validate_dataset_columns

st.set_page_config(
    page_title="Soil AI | Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_database()

st.markdown("""
<style>
.main-title {font-size: 2.3rem; font-weight: 800;}
.subtitle {font-size: 1.05rem; color: #6b7280;}
.card {padding: 1rem; border-radius: 14px; border: 1px solid #e5e7eb;
       background: #ffffff; margin-bottom: 1rem;}
.crop {font-size: 1.4rem; font-weight: 700;}
.small {color: #6b7280; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🌱 Soil Crop AI")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Soil Analysis", "Crop Recommendation", "XAI Explanation",
     "Model Performance", "Prediction History", "About"]
)

def input_form():
    st.subheader("Soil Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n = st.number_input("Nitrogen (kg/ha)", min_value=0.0, value=50.0)
    with c2:
        p = st.number_input("Phosphorus (kg/ha)", min_value=0.0, value=50.0)
    with c3:
        k = st.number_input("Potassium (kg/ha)", min_value=0.0, value=50.0)
    with c4:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)

    st.subheader("Environmental Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.number_input("Temperature (°C)", value=25.0)
    with c2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
    with c3:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)

    return {
        "N": n, "P": p, "K": k, "temperature": temperature,
        "humidity": humidity, "ph": ph, "rainfall": rainfall
    }

if page == "Home":
    st.markdown('<div class="main-title">🌱 AI-Based Soil Analysis & Intelligent Crop Recommendation</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Machine Learning + Explainable AI for Data-Driven Agriculture</div>',
                unsafe_allow_html=True)
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("ML", "Classification")
    c2.metric("XAI", "SHAP")
    c3.metric("Database", "SQLite")
    st.markdown("""
    ### Project Overview
    This application accepts soil and environmental conditions, predicts suitable crops,
    ranks the Top-3 model recommendations, explains the prediction with SHAP, and stores
    prediction history locally.

    ### Core workflow
    **User Input → Validation → Trained ML Pipeline → Crop Prediction → Top-3 Ranking →
    SHAP Explanation → Soil Health → SQLite History**

    ### Important
    Model probabilities are model-estimated probabilities/confidence values, not guarantees
    of farming success. Actual agricultural decisions should consider laboratory soil tests,
    local climate, agronomist guidance, and crop-specific requirements.
    """)

elif page == "Soil Analysis":
    st.title("🔬 Soil Analysis")
    values = input_form()
    if st.button("Analyze Soil", type="primary"):
        try:
            result = analyze_soil(**values)
            st.subheader("Soil Health Summary")
            cols = st.columns(4)
            for col, key in zip(cols, ["N", "P", "K", "ph"]):
                col.metric(key, result["status"][key])
            st.metric("Soil Health Score", f'{result["score"]:.1f}/100')
            st.info(result["disclaimer"])
        except ValueError as e:
            st.error(str(e))

elif page == "Crop Recommendation":
    st.title("🌾 Crop Recommendation")
    if not model_available():
        st.error("Trained model not found. Run: python src/train_model.py")
    values = input_form()
    if st.button("Recommend Crops", type="primary"):
        try:
            result = predict_top3(**values)
            st.success(f'Predicted Crop: {result["predicted_crop"]}')
            cols = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            for col, medal, item in zip(cols, medals, result["top3"]):
                col.metric(f"{medal} {item['crop']}", f"{item['probability']:.2%}")
            st.caption("These are model-estimated probabilities, not crop-success probabilities.")

            health = analyze_soil(**values)
            st.subheader("Soil Health")
            st.metric("Score", f'{health["score"]:.1f}/100')

            explanation = explain_prediction(values)
            st.subheader("Why did the model recommend this crop?")
            st.write(explanation["plain_english"])
            save_prediction(values, result["predicted_crop"], result["top3"])
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "XAI Explanation":
    st.title("🧠 Explainable AI")
    if not model_available():
        st.warning("Train the model first.")
    else:
        st.subheader("Global Feature Importance")
        try:
            imp = global_feature_importance()
            st.bar_chart(imp.set_index("feature"))
        except Exception as e:
            st.error(f"Could not load global SHAP importance: {e}")

        st.subheader("Explain a New Prediction")
        values = input_form()
        if st.button("Explain Prediction", type="primary"):
            try:
                exp = explain_prediction(values)
                st.success(f'Predicted crop: {exp["predicted_crop"]}')
                st.dataframe(exp["contributions"], use_container_width=True)
                st.write(exp["plain_english"])
            except Exception as e:
                st.error(f"XAI failed: {e}")

elif page == "Model Performance":
    st.title("📊 Model Performance")
    meta_path = ROOT / "models" / "model_metadata.json"
    if not meta_path.exists():
        st.warning("No model metadata found. Run the training script first.")
    else:
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        st.write("**Selected model:**", meta.get("selected_model"))
        st.write("**Classes:**", meta.get("num_classes"))
        st.write("**Training samples:**", meta.get("training_samples"))
        st.write("**Testing samples:**", meta.get("testing_samples"))
        if meta.get("model_comparison"):
            import pandas as pd
            st.dataframe(pd.DataFrame(meta["model_comparison"]), use_container_width=True)
        if meta.get("confusion_matrix"):
            import plotly.express as px
            fig = px.imshow(meta["confusion_matrix"], text_auto=True,
                            title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Prediction History":
    st.title("🗃️ Prediction History")
    history = get_prediction_history()
    if history.empty:
        st.info("No predictions have been stored yet.")
    else:
        st.dataframe(history, use_container_width=True)

elif page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    **AI-Based Soil Analysis and Intelligent Crop Recommendation Using Machine Learning
    and Explainable AI**

    Technology stack:
    - Python 3.12
    - Pandas / NumPy
    - Scikit-learn
    - XGBoost (optional)
    - SHAP
    - Streamlit
    - Plotly / Matplotlib
    - Joblib
    - SQLite

    **Disclaimer:** This is a decision-support system. It is not a replacement for
    professional agricultural advice or laboratory soil testing.
    """)