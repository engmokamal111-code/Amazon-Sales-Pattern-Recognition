import streamlit as st
import joblib
import numpy as np

# 1. تحميل الموديلات المحدثة
# تأكد أن هذه الملفات موجودة في نفس المستودع (Repository) على GitHub
try:
    model = joblib.load('amazon_svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_model.pkl')
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure .pkl files are uploaded to GitHub.")

# 2. إعدادات الصفحة
st.set_page_config(page_title="Amazon Sales Predictor", page_icon="🚀")
st.title("🚀 Amazon Sales Performance Predictor")
st.write("This app uses a Balanced SVM model with PCA to predict product performance.")

# 3. واجهة المدخلات (مرتبة حسب ميزات التدريب)
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Original Price", min_value=0.0, value=500.0, step=10.0)
    discount_percent = st.slider("Discount Percentage (%)", 0, 100, 25)
    rating = st.slider("Customer Rating (1-5)", 1.0, 5.0, 4.5, step=0.1)

with col2:
    quantity_sold = st.number_input("Quantity Sold", min_value=0, value=1000, step=50)
    # ملاحظة: تأكد من إدخال رقم الفئة الذي استخدمته في التدريب
    category_encoded = st.number_input("Product Category ID", min_value=0, max_value=20, value=1)

# 4. منطق التوقع
if st.button("Predict Performance"):
    # تجهيز المصفوفة بالترتيب الصحيح لـ Features
    # Order: ['price', 'discount_percent', 'rating', 'quantity_sold', 'product_category_encoded']
    input_features = np.array([[price, discount_percent, rating, quantity_sold, category_encoded]])
    
    # تحويل البيانات باستخدام السكيلر والـ PCA المحدثين
    input_scaled = scaler.transform(input_features)
    input_pca = pca.transform(input_scaled)
    
    # إجراء التوقع
    prediction = model.predict(input_pca)[0]
    
    # عرض النتيجة بتنسيق لوني جذاب
    st.markdown("---")
    if prediction == 'High' or prediction == 0: # التعامل مع الاحتمالين (نص أو رقم)
        st.success(f"### Result: High Performance 🌟")
        st.balloons()
    elif prediction == 'Medium' or prediction == 2:
        st.warning(f"### Result: Medium Performance 📊")
    else:
        st.error(f"### Result: Low Performance 📉")

# 5. معلومات إضافية للدكتور
with st.expander("Technical Details"):
    st.write("""
    - **Preprocessing:** Standard Scaler
    - **Dimensionality Reduction:** Principal Component Analysis (PCA) - 2 Components
    - **Model:** Support Vector Machine (SVM) with RBF Kernel
    - **Optimization:** SMOTE (Synthetic Minority Over-sampling Technique) for class balancing.
    """)
