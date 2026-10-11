import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not GEMINI_API_KEY:
    raise ValueError(
        "Google Gemini API key is not configured."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


def get_agriculture_advice(soil_data, predicted_crop, explanation):
    """
    Generate agricultural advice using Gemini AI.

    Parameters:
        soil_data: Soil parameters entered by the user
        predicted_crop: Crop predicted by ML model
        explanation: XAI explanation of the prediction

    Returns:
        Gemini-generated agricultural advice
    """

    prompt = f"""
You are an agricultural AI assistant.

The following information comes from a machine-learning
soil analysis and crop recommendation system.

Soil data:
{soil_data}

Machine-learning predicted crop:
{predicted_crop}

Explainable AI model explanation:
{explanation}

Provide the response in the following sections:

1. Why this crop is suitable
2. Soil condition interpretation
3. Basic cultivation suggestions
4. Important precautions

Important rules:
- Do not invent soil measurements.
- Use only the soil data provided above.
- Clearly distinguish ML model predictions from general agricultural advice.
- Do not claim that the recommendation is guaranteed.
- Keep the explanation simple and understandable.
- If information is insufficient, clearly say so.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Gemini AI error: {str(e)}"
