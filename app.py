import streamlit as st
import joblib
import numpy as np

# 1. إعدادات الصفحة (تظهر في تبويب المتصفح)
st.set_page_config(page_title="Amazon Predictor Pro", page_icon="📊", layout="centered")

# 2. تحميل الموديلات
@st.cache_resource # لتسريع الموقع ومنع إعادة تحميل الموديل مع كل ضغطة زر
def load_models():
    model = joblib.load('amazon_svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_model.pkl')
    return model, scaler, pca

try:
    model, scaler, pca = load_models()
except Exception as e:
    st.error("⚠️ Error: Missing model files. Please upload .pkl files to GitHub.")
    st.stop()

# 3. واجهة المستخدم
st.title("🚀 Amazon Sales Performance Predictor")
st.markdown("""
هذا النظام يستخدم الذكاء الاصطناعي (SVM) للتنبؤ بأداء المنتجات بناءً على أنماط مبيعات أمازون.
""")

st.info("قم بإدخال تفاصيل المنتج أدناه للحصول على التوقع.")

# تقسيم المدخلات لعمودين لشكل أكثر احترافية
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Original Price (السعر الأصلي)", min_value=0.0, value=500.0)
    discount_percent = st.slider("Discount % (نسبة الخصم)", 0, 100, 25)
    rating = st.slider("Customer Rating (التقييم)", 1.0, 5.0, 4.5, step=0.1)

with col2:
    quantity_sold = st.number_input("Quantity Sold (الكمية المباعة)", min_value=0, value=1000)
    category_encoded = st.number_input("Category ID (رقم الفئة)", min_value=0, max_value=20, value=1)

# 4. منطق التوقع (Prediction Logic)
if st.button("Predict Performance"):
    # ترتيب الميزات تماماً كما في الكولاب:
    # [price, discount_percent, rating, quantity_sold, product_category_encoded]
    input_data = np.array([[price, discount_percent, rating, quantity_sold, category_encoded]])
    
    # تحويل البيانات (Scaling + PCA)
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    
    # التوقع
    prediction = model.predict(input_pca)[0]
    
    st.markdown("### Result Analysis:")
    
    # ربط الرقم بالفئة الصحيحة بناءً على الموديل:
    # 0 -> High | 1 -> Low | 2 -> Medium
    
    if prediction == 0 or str(prediction) == 'High':
        st.success("## [High Performance] 🌟🌟🌟")
        st.balloons()
        st.write("المنتج يحقق أداءً ممتازاً ومبيعات مرتفعة جداً.")
        
    elif prediction == 2 or str(prediction) == 'Medium':
        st.warning("## [Medium Performance] 📊")
        st.write("أداء المنتج مستقر ومتوسط في السوق.")
        
    else: # في حالة كانت النتيجة 1 (Low)
        st.error("## [Low Performance] 📉")
        st.write("أداء المنتج ضعيف ويحتاج لتحسين السعر أو التقييم.")

# 5. تذييل الصفحة (Footer)
st.markdown("---")
st.caption("Developed for Pattern Recognition Course Project | SVM & PCA Implementation")
