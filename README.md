# Lung Cancer Detection using Deep Learning

## Overview

This project is an AI-based medical imaging system that detects lung cancer from CT scan images using a Convolutional Neural Network (CNN).
The system analyzes lung CT scan images and classifies them into three categories:

* Normal
* Benign (non-cancerous tumor)
* Malignant (cancerous tumor)

A web application built with Streamlit allows users to upload CT scan images and receive predictions from the trained deep learning model.

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Streamlit

---

## Project Structure

lung-cancer-detection
│
├── dataset/
│   ├── normal cases/
│   ├── benign cases/
│   └── malignant cases/
│
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md

---

## Features

* Deep learning model for lung cancer detection
* CT scan image preprocessing using OpenCV
* Classification into Normal, Benign, and Malignant cases
* Streamlit web interface for uploading CT scan images
* Real-time prediction results

---

## Installation

### 1. Clone the repository

git clone https://github.com/your-username/lung-cancer-detection-ai.git

cd lung-cancer-detection-ai

### 2. Install required libraries

pip install -r requirements.txt

---

## Train the Model

Run the following command to train the CNN model:

python train_model.py

This will generate the trained model file:

lung_cancer_model.h5
https://drive.google.com/file/d/1jFzKMIzz82r_AZ-T-eJOVxI2b4nQIVbJ/view?usp=drive_link

---

## Run the Web Application

Start the Streamlit app:

streamlit run app.py

Then open the local link shown in the terminal (usually http://localhost:8501).

Upload a CT scan image to detect whether it is Normal, Benign, or Malignant.

---

## Dataset

This project uses a lung CT scan dataset containing three classes:

* Normal cases
* Benign cases
* Malignant cases

Example dataset: IQ-OTH/NCCD Lung Cancer Dataset.

---

## How It Works

1. User uploads a lung CT scan image.
2. The image is resized and normalized.
3. The trained CNN model analyzes the image.
4. The model predicts one of the three classes:

   * Normal
   * Benign
   * Malignant

---

## Disclaimer

This project is developed for educational and research purposes only and should not be used for real medical diagnosis.

---

## Future Improvements

* Improve model accuracy using Transfer Learning (ResNet50 / EfficientNet)
* Add Grad-CAM visualization to highlight tumor regions
* Deploy the application publicly for online access
* Improve dataset size for better generalization

---

## Author

Manoj Badhan
BTech AI & Robotics Engineering

