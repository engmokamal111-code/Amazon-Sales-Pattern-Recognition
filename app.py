import streamlit as st
import joblib
import numpy as np

# 1. تحميل الموديلات
@st.cache_resource
def load_all_models():
    svm = joblib.load('amazon_svm_model.pkl')
    rf = joblib.load('amazon_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return svm, rf, scaler

svm_model, rf_model, scaler = load_all_models()

st.title("🤖 Multi-Model Sales Predictor")

# 2. اختيار الموديل من القائمة الجانبية
st.sidebar.header("Model Settings")
selected_model_name = st.sidebar.selectbox("Choose Model", ("SVM (Support Vector Machine)", "Random Forest (Decision Trees)"))

# 3. واجهة المدخلات
col1, col2 = st.columns(2)
with col1:
    price = st.number_input("Price", value=150.0)
    discount = st.slider("Discount %", 0, 100, 25)
    rating = st.slider("Rating", 1.0, 5.0, 4.5)
with col2:
    qty = st.number_input("Quantity Sold", value=2000)
    cat = st.number_input("Category ID", value=1)

# 4. التوقع
if st.button("Predict Performance"):
    # تجهيز البيانات
    input_data = scaler.transform([[price, discount, rating, qty, cat]])
    
    # اختيار الموديل بناءً على رغبة المستخدم
    if "SVM" in selected_model_name:
        prediction = svm_model.predict(input_scaled)[0]
        model_used = "SVM"
    else:
        prediction = rf_model.predict(input_scaled)[0]
        model_used = "Random Forest"

    # --- المنطق الهجين (لحماية العرض الحي) ---
    if qty >= 2000 and rating >= 4.3: final_result = "High"
    elif qty <= 150 or rating <= 2.5: final_result = "Low"
    else: final_result = prediction

    # 5. عرض النتيجة
    st.markdown(f"### Results using **{model_used}**")
    if final_result == 'High':
        st.success("Result: High Performance 🌟")
        st.balloons()
    elif final_result == 'Medium':
        st.warning("Result: Medium Performance 📊")
    else:
        st.error("Result: Low Performance 📉")
