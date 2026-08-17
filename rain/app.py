import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Rain Predictor", page_icon="🌧️", layout="centered"
)


# -------------------------------------------------
# Load trained model (Pipeline with preprocessing included)
# -------------------------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "rain_model.pkl")
    locations_path = os.path.join(BASE_DIR, "locations.pkl")

    model = joblib.load(model_path)
    locations = joblib.load(locations_path)
    return model, locations


model, locations = load_model()

st.title("🌧️ Rain Tomorrow Predictor")
st.write(
    "أدخل بيانات الطقس اليوم للتنبؤ باحتمالية هطول المطر غدًا (بيانات"
    " أستراليا)."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox(
        "المدينة (Location)",
        locations,
        index=locations.index("Sydney") if "Sydney" in locations else 0,
    )
    rain_today = st.selectbox("هل مطرت اليوم؟ (RainToday)", ["No", "Yes"])
    min_temp = st.slider("أقل درجة حرارة (MinTemp °C)", -10, 35, 15)
    max_temp = st.slider("أعلى درجة حرارة (MaxTemp °C)", 0, 50, 25)
    rainfall = st.number_input(
        "كمية المطر اليوم (Rainfall mm)",
        min_value=0.0,
        max_value=300.0,
        value=0.0,
        step=0.5,
    )
    wind_gust = st.slider("أقصى سرعة رياح (WindGustSpeed km/h)", 0, 130, 40)

with col2:
    humidity9 = st.slider("الرطوبة الساعة 9 صباحًا (%)", 0, 100, 60)
    humidity3 = st.slider("الرطوبة الساعة 3 عصرًا (%)", 0, 100, 50)
    pressure9 = st.slider("الضغط الجوي 9 صباحًا (hPa)", 980, 1040, 1015)
    pressure3 = st.slider("الضغط الجوي 3 عصرًا (hPa)", 980, 1040, 1012)
    temp9 = st.slider("درجة الحرارة 9 صباحًا (°C)", -10, 45, 18)
    temp3 = st.slider("درجة الحرارة 3 عصرًا (°C)", -10, 45, 23)

wind9 = st.slider("سرعة الرياح 9 صباحًا (km/h)", 0, 100, 15)
wind3 = st.slider("سرعة الرياح 3 عصرًا (km/h)", 0, 100, 18)

st.divider()

if st.button("🔮 تنبأ بالمطر غدًا", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([
        {
            "MinTemp": min_temp,
            "MaxTemp": max_temp,
            "Rainfall": rainfall,
            "WindGustSpeed": wind_gust,
            "WindSpeed9am": wind9,
            "WindSpeed3pm": wind3,
            "Humidity9am": humidity9,
            "Humidity3pm": humidity3,
            "Pressure9am": pressure9,
            "Pressure3pm": pressure3,
            "Temp9am": temp9,
            "Temp3pm": temp3,
            "RainToday": rain_today,
            "Location": location,
        }
    ])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(
            f"🌧️ الأرجح إنها **هتمطر بكرة**  |  احتمالية المطر: {probability:.1%}"
        )
    else:
        st.info(
            f"☀️ الأرجح إنها **مش هتمطر بكرة**  |  احتمالية المطر:"
            f" {probability:.1%}"
        )

    st.progress(float(probability))

    with st.expander("عرض البيانات المُدخلة"):
        st.dataframe(input_df)

st.divider()
st.caption(
    "مبني باستخدام Logistic Regression / KNN مع Scikit-learn Pipeline. المشروع"
    " النهائي - Rain Prediction (Australia)."
)
