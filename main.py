import streamlit as st
from super_image import EdsrModel, ImageLoader
from PIL import Image
import numpy as np
import torch

st.set_page_config(page_title="Super Resolution App", layout="centered")
st.title("🎯 Super Resolution App")

@st.cache_resource
def load_model():
    return EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a Low-Resolution Image",
    type=["jpg", "jpeg", "JPG", "JPEG", "png"]
)

if uploaded_file:
    lr_image = Image.open(uploaded_file).convert("RGB")
    st.image(lr_image, caption="Low-Resolution Image", use_column_width=True)

    with st.spinner("Upscaling..."):
        inputs = ImageLoader.load_image(lr_image)
        preds = model(inputs)

        # Convert to numpy if tensor
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        # Remove batch dimension
        if preds.ndim == 4:
            preds = preds[0]

        # Convert CHW to HWC
        if preds.shape[0] == 3:
            preds = np.transpose(preds, (1, 2, 0))

        # Normalize to [0, 255] to avoid black images
        min_val = preds.min()
        max_val = preds.max()
        hr_array = (preds - min_val) / (max_val - min_val + 1e-8)
        hr_array = (hr_array * 255).astype(np.uint8)

        # Convert to PIL image
        hr_image = Image.fromarray(hr_array)

    st.image(hr_image, caption="High-Resolution Output", use_column_width=True)
    st.success("Upscaling Completed!")
