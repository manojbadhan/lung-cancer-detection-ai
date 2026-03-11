import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import os

# -------- Model Download Section --------

model_path = "lung_cancer_model.h5"

# Download model from Google Drive if it does not exist

if not os.path.exists(model_path):
file_id = "1jFzKMIzz82r_AZ-T-eJOVxI2b4nQIVbJ"   # your drive file id
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, model_path, quiet=False)

# -------- Load Model --------

model = load_model(model_path, compile=False)

classes = ["Normal", "Benign", "Malignant"]

st.title("Lung Cancer Detection AI By Manoj Badhan")
st.write("Upload a lung CT scan image to detect cancer.")

uploaded_file = st.file_uploader("Upload CT Scan", type=["jpg","png","jpeg"])

if uploaded_file is not None:

```
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)

st.image(img, caption="Uploaded CT Scan", use_container_width=True)

img = cv2.resize(img,(128,128))
img = img/255.0
img = np.reshape(img,(1,128,128,3))

prediction = model.predict(img)

result = classes[np.argmax(prediction)]

st.subheader("Prediction Result")

if result == "Malignant":
    st.error("Malignant (Cancer Detected)")
elif result == "Benign":
    st.warning("Benign Tumor Detected")
else:
    st.success("Normal Lung")
```
