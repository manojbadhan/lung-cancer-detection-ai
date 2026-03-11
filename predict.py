import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("lung_cancer_model.h5")

img = cv2.imread("test.png")
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.reshape(img,(1,128,128,3))

prediction = model.predict(img)

classes = ["Normal","Benign","Malignant"]

print("Prediction:", classes[np.argmax(prediction)])