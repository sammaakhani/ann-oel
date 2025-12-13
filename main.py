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

        # Convert preds to numpy if it's a tensor
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        # Remove batch dimension if present
        if preds.ndim == 4:
            preds = preds[0]  # from (1, C, H, W) to (C, H, W)

        # Convert CHW to HWC
        if preds.shape[0] == 3:
            preds = np.transpose(preds, (1, 2, 0))

        # Scale to uint8
        hr_array = np.clip(preds, 0, 1) * 255 if preds.max() <= 1.0 else preds
        hr_array = hr_array.astype(np.uint8)

        # Convert to PIL
        hr_image = Image.fromarray(hr_array)

    st.image(hr_image, caption="High-Resolution Output", use_column_width=True)
    st.success("Upscaling Completed!")
