import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# -------------------------------------------------
# Load trained model (Pipeline with preprocessing included)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write("أدخل بيانات الراكب للتنبؤ باحتمالية نجاته من كارثة تيتانيك.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("درجة التذكرة (Pclass)", [1, 2, 3], index=2,
                           help="1 = درجة أولى، 2 = درجة ثانية، 3 = درجة ثالثة")
    sex = st.selectbox("الجنس (Sex)", ["male", "female"])
    age = st.slider("العمر (Age)", 0, 80, 30)
    embarked = st.selectbox("ميناء الصعود (Embarked)", ["S", "C", "Q"],
                             help="S = Southampton, C = Cherbourg, Q = Queenstown")

with col2:
    sibsp = st.number_input("عدد الإخوة/الزوج على متن السفينة (SibSp)", min_value=0, max_value=8, value=0)
    parch = st.number_input("عدد الأبناء/الآباء على متن السفينة (Parch)", min_value=0, max_value=6, value=0)
    fare = st.number_input("سعر التذكرة (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

st.divider()

if st.button("🔮 تنبأ بالنجاة", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ الراكب على الأرجح **نجا**  |  احتمالية النجاة: {probability:.1%}")
    else:
        st.error(f"❌ الراكب على الأرجح **لم ينجُ**  |  احتمالية النجاة: {probability:.1%}")

    st.progress(float(probability))

    with st.expander("عرض البيانات المُدخلة"):
        st.dataframe(input_df)

st.divider()
st.caption("مبني باستخدام Logistic Regression / KNN مع Scikit-learn Pipeline. المشروع النهائي - Titanic Classification.")
