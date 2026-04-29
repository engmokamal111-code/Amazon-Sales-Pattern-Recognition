import streamlit as st
import joblib
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Amazon Sales Predictor", page_icon="🛍️", layout="centered")

# 2. تحميل الموديلات (بدون PCA)
@st.cache_resource
def load_assets():
    try:
        # تحميل الموديل والسكيلر المحدثين
        model = joblib.load('amazon_svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None

model, scaler = load_assets()

# 3. واجهة المستخدم
st.title("🛒 Amazon Sales Performance Predictor")
st.markdown("توقع أداء مبيعات منتجات أمازون باستخدام **SVM Model** المحدث (بدون PCA لضمان أعلى دقة).")

if model is None:
    st.warning("الرجاء التأكد من رفع ملفات `amazon_svm_model.pkl` و `scaler.pkl` على GitHub.")
    st.stop()

# تصميم المدخلات في أعمدة
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Original Price (السعر)", min_value=0.0, value=150.0)
    discount = st.slider("Discount % (الخصم)", 0, 100, 25)
    rating = st.slider("Rating (التقييم من 5)", 1.0, 5.0, 4.2, step=0.1)

with col2:
    qty = st.number_input("Quantity Sold (الكمية المباعة)", min_value=0, value=800)
    # ملاحظة: تأكد من إدخال رقم الفئة الذي استخدمته في التدريب
    cat = st.number_input("Category ID (رقم الفئة)", min_value=0, max_value=100, value=1)

# 4. منطق التوقع
if st.button("Predict Performance"):
    # تجهيز البيانات بنفس الترتيب: [price, discount_percent, rating, quantity_sold, category]
    input_data = np.array([[price, discount, rating, qty, cat]])
    
    # تحويل البيانات باستخدام السكيلر فقط
    input_scaled = scaler.transform(input_data)
    
    # التوقع (سيعيد الموديل نصاً: 'High' أو 'Medium' أو 'Low')
    prediction = model.predict(input_scaled)[0]
    
    st.markdown("---")
    st.subheader("النتيجة المتوقعة:")

    # عرض النتيجة بناءً على المخرج النصي للموديل
    if prediction == 'High':
        st.success(f"### Result: {prediction} Performance 🌟")
        st.balloons()
        st.write("أداء ممتاز! المنتج يحقق مبيعات عالية جداً مقارنة بالمنافسين.")
        
    elif prediction == 'Medium':
        st.warning(f"### Result: {prediction} Performance 📊")
        st.write("أداء متوسط. المنتج مستقر في السوق ولكن يمكن تحسينه بالخصومات.")
        
    else: # Low
        st.error(f"### Result: {prediction} Performance 📉")
        st.write("أداء منخفض. نوصي بمراجعة السعر أو تحسين جودة المنتج لرفع التقييم.")

# 5. تذييل الصفحة
st.info("نصيحة للمناقشة: تم إلغاء الـ PCA للحفاظ على كامل تباين البيانات (Full Variance)، مما جعل الموديل أكثر قدرة على الفصل بين الفئات.")
