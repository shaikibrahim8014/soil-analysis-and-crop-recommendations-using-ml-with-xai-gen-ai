# AI-Based Soil Analysis and Intelligent Crop Recommendation

A B.Tech final-year project using Python, Machine Learning, SHAP Explainable AI,
Streamlit and SQLite.

## Features

- Manual soil/environment input
- Input validation
- Multiple classification models
- Actual model comparison
- Random Forest tuning when selected
- Top-3 crop recommendations
- Model-estimated probabilities
- SHAP-based explanation
- Simplified soil-health analysis
- SQLite prediction history
- Streamlit dashboard
- Model performance page

## Dataset

Place your real dataset at:

`data/soil_crop_dataset.csv`

Required columns:

`N,P,K,temperature,humidity,ph,rainfall,label`

Do not use the header-only CSV as a training dataset. Replace it with your actual
dataset containing multiple crop classes.

## Installation

Python 3.12 is recommended.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python src/train_model.py
```

This creates:

- `models/crop_model.pkl`
- `models/model_metadata.json`

## Run

```bash
streamlit run app.py
```

## Architecture

User → Streamlit → Validation → ML Pipeline → Prediction → Top-3 →
SHAP → Soil Health → SQLite History

## Important interpretation

The displayed probability is the model's estimated class probability. It is not
a probability that the crop will successfully grow.

The simplified soil-health thresholds and score are project rules and are not
universal agronomic standards.

## Testing

After training:

```bash
pytest
```

## Limitations

- Dataset quality directly affects model quality.
- Crop recommendation is based on the available features.
- Weather, local agronomy, irrigation, seed variety and disease conditions are
  not fully represented.
- SHAP explains model behavior; it does not prove agricultural causation.

## Disclaimer

This system provides data-driven decision support and is not a replacement for
professional agricultural advice or laboratory soil testing.
