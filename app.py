import streamlit as st
import joblib
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Amazon Performance Predictor", page_icon="🛒", layout="centered")

# 2. تحميل الموديلات (الموديل والسكيلر بدون PCA)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('amazon_svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

# 3. واجهة المستخدم
st.title("🛒 Amazon Sales Performance Predictor")
st.markdown("""
هذا النظام يدمج بين **ذكاء الآلة (SVM)** و **المنطق الإحصائي** للتنبؤ بأداء المنتجات.
""")

if model is None:
    st.error("⚠️ خطأ: لم يتم العثور على ملفات الموديل. تأكد من رفع `amazon_svm_model.pkl` و `scaler.pkl` على GitHub.")
    st.stop()

# تصميم المدخلات
st.info("أدخل بيانات المنتج للحصول على تحليل الأداء:")
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Original Price ($)", min_value=0.0, value=150.0)
    discount = st.slider("Discount %", 0, 100, 25)
    rating = st.slider("Customer Rating", 1.0, 5.0, 4.2, step=0.1)

with col2:
    qty = st.number_input("Quantity Sold", min_value=0, value=800)
    cat = st.number_input("Category ID", min_value=0, max_value=100, value=1)

# 4. منطق التوقع (The Hybrid Engine)
if st.button("Predict Performance"):
    # تجهيز البيانات للسكيلر والموديل
    input_data = np.array([[price, discount, rating, qty, cat]])
    input_scaled = scaler.transform(input_data)
    
    # الحصول على توقع الموديل الأصلي
    raw_prediction = model.predict(input_scaled)[0]
    
    # --- Logic Override (لضمان استجابة النظام للأرقام الاستثنائية) ---
    # هذه القواعد تضمن أن الأرقام العالية جداً تظهر كـ High والأرقام الضعيفة جداً تظهر كـ Low
    if qty >= 2000 and rating >= 4.3:
        final_result = "High"
    elif qty <= 150 or rating <= 2.5:
        final_result = "Low"
    else:
        # إذا كانت الأرقام في المنطقة المتوسطة، نعتمد كلياً على قرار الموديل
        final_result = raw_prediction

    # 5. عرض النتيجة النهائية
    st.markdown("---")
    if final_result == 'High':
        st.success(f"### Result: High Performance 🌟🌟🌟")
        st.balloons()
        st.write("**التحليل:** المنتج يتفوق بوضوح في السوق مع مبيعات قوية وثقة عالية من العملاء.")
    elif final_result == 'Medium':
        st.warning(f"### Result: Medium Performance 📊")
        st.write("**التحليل:** أداء المنتج مستقر ومتوسط. هناك فرصة لزيادة المبيعات عبر تحسين الخصومات.")
    else:
        st.error(f"### Result: Low Performance 📉")
        st.write("**التحليل:** الأداء الحالي ضعيف. قد يحتاج المنتج لتعديل السعر أو تحسين الجودة لرفع التقييم.")

    # قسم تقني للمناقشة
    with st.expander("🔍 تفاصيل تقنية (Technical Insights)"):
        st.write(f"**Model Raw Output:** {raw_prediction}")
        st.write(f"**Final Decision Logic:** Hybrid (SVM + Expert Rules)")
        st.write("**Algorithm:** Support Vector Machine (RBF Kernel)")

# 6. تذييل الصفحة
st.markdown("---")
st.caption("Machine Learning Project - Amazon Products Analysis")
