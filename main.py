import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

st.set_page_config(page_title="Super Resolution App", layout="centered")
st.title("🎯 Image Super Resolution App (Pretrained Model)")

@st.cache_resource
def load_model():
    model = torch.hub.load(
        "sanghyun-son/EDSR-PyTorch",
        "edsr",
        scale=2,
        pretrained=True
    )
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a Low-Resolution Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    lr_image = Image.open(uploaded_file).convert("RGB")
    st.image(lr_image, caption="Low-Resolution Image", use_column_width=True)

    with st.spinner("Upscaling image..."):
        lr_tensor = ToTensor()(lr_image).unsqueeze(0)

        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        sr_tensor = sr_tensor.squeeze(0).clamp(0, 1)
        hr_image = ToPILImage()(sr_tensor)

    st.image(hr_image, caption="High-Resolution Output", use_column_width=True)
    st.success("Upscaling Completed!")
