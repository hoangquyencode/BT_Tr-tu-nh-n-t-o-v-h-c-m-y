import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Tắt GPU (tránh lỗi máy yếu)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.set_page_config(
    page_title="Dog vs Cat AI",
    page_icon="🐶",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>
    🐶🐱 AI Phân Loại Chó & Mèo
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Chọn cách nhập ảnh bên dưới 👇")

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_cat_model.h5")

model = load_model()

# ===== CHỌN CHẾ ĐỘ =====
option = st.radio(
    "Nguồn ảnh:",
    ("📁 Upload ảnh", "📷 Dùng Webcam")
)

image = None

if option == "📁 Upload ảnh":
    uploaded_file = st.file_uploader(
        "Chọn ảnh...",
        type=["jpg", "png", "jpeg"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

else:
    camera_image = st.camera_input("Chụp ảnh")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# ===== DỰ ĐOÁN =====
if image is not None:
    st.image(image, caption="Ảnh đầu vào", use_column_width=True)

    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Dự đoán"):
        with st.spinner("Đang phân tích..."):
            prediction = model.predict(img_array)[0][0]

        cat_prob = (1 - prediction) * 100
        dog_prob = prediction * 100

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🐱 Mèo", f"{cat_prob:.2f}%")

        with col2:
            st.metric("🐶 Chó", f"{dog_prob:.2f}%")

        st.markdown("### 📊 Xác suất")

        st.write("Chó")
        st.progress(int(dog_prob))

        st.write("Mèo")
        st.progress(int(cat_prob))

        st.markdown("---")

        if prediction >= 0.5:
            st.success("👉 Kết luận: Đây là CHÓ 🐶")
        else:
            st.success("👉 Kết luận: Đây là MÈO 🐱")

        # Debug nếu cần
        st.caption(f"Raw prediction value: {prediction:.4f}")