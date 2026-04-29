import streamlit as st
import joblib
import numpy as np

# 1. تحميل الموديلات
model = joblib.load('amazon_svm_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')

# 2. تصميم واجهة الموقع
st.title("🚀 Amazon Sales Performance Predictor")
st.write("Enter product details to predict its market performance category.")

# 3. إنشاء خانات الإدخال
price = st.number_input("Original Price", value=500.0)
discount = st.slider("Discount Percentage (%)", 0, 100, 10)
rating = st.slider("Customer Rating (1-5)", 1.0, 5.0, 4.0)
quantity = st.number_input("Quantity Sold", value=100)
category = st.selectbox("Product Category ID", [0, 1, 2, 3, 4, 5]) # بناءً على الـ Encoding بتاعك

# 4. زر التوقع
if st.button("Predict Performance"):
    # تجهيز البيانات
    input_data = np.array([[price, discount, rating, quantity, category]])
    
    # التحويل (Preprocessing)
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    
    # التوقع
    prediction = model.predict(input_pca)[0]
    
    # عرض النتيجة بشكل جمالي
    if prediction == 'High':
        st.success(f"Result: {prediction} Performance 🌟")
    elif prediction == 'Medium':
        st.warning(f"Result: {prediction} Performance 📊")
    else:
        st.error(f"Result: {prediction} Performance 📉")