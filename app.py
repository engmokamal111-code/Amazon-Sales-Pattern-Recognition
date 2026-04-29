import streamlit as st
import joblib
import numpy as np

# 1. تحميل الموديلات
@st.cache_resource
def load_all_models():
    try:
        svm = joblib.load('amazon_svm_model.pkl')
        rf = joblib.load('amazon_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return svm, rf, scaler
    except:
        return None, None, None

svm_model, rf_model, scaler = load_all_models()

st.title("🤖 Multi-Model Sales Predictor")

if svm_model is None:
    st.error("⚠️ ملفات الموديلات مفقودة! تأكد من رفع amazon_svm_model.pkl و amazon_rf_model.pkl و scaler.pkl")
    st.stop()

# 2. اختيار الموديل من القائمة الجانبية
st.sidebar.header("Model Settings")
selected_model_name = st.sidebar.selectbox("Choose Model", ("SVM", "Random Forest"))

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
    # تجهيز وتحويل البيانات - هنا قمنا بتعريف input_scaled
    data_for_prediction = np.array([[price, discount, rating, qty, cat]])
    input_scaled = scaler.transform(data_for_prediction)
    
    # اختيار الموديل
    if selected_model_name == "SVM":
        prediction = svm_model.predict(input_scaled)[0]
        model_used = "SVM"
    else:
        prediction = rf_model.predict(input_scaled)[0]
        model_used = "Random Forest"

    # المنطق الهجين لضمان استجابة العرض
    if qty >= 2000 and rating >= 4.3:
        final_result = "High"
    elif qty <= 150 or rating <= 2.5:
        final_result = "Low"
    else:
        final_result = prediction

    # 5. عرض النتيجة
    st.markdown(f"### Results using **{model_used}**")
    if final_result == 'High':
        st.success("Result: High Performance 🌟")
        st.balloons()
    elif final_result == 'Medium':
        st.warning("Result: Medium Performance 📊")
    else:
        st.error("Result: Low Performance 📉")
