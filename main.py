import streamlit as st
from super_image import EdsrModel, ImageLoader
from PIL import Image

st.set_page_config(page_title="Super Resolution App", layout="centered")
st.title("🎯 Super Resolution App")

@st.cache_resource
def load_model():
    return EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)

model = load_model()

uploaded_file = st.file_uploader("Upload a Low-Resolution Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    lr_image = Image.open(uploaded_file).convert("RGB")
    st.image(lr_image, caption="Low-Resolution Image", use_column_width=True)

    with st.spinner("Upscaling..."):
        inputs = ImageLoader.load_image(lr_image)
        preds = model(inputs)
        hr_image = ImageLoader.to_pil(preds)

    st.image(hr_image, caption="High-Resolution Output", use_column_width=True)
    st.success("Upscaling Completed!")
