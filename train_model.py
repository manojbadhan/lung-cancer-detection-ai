import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

data = []
labels = []

dataset_path = "dataset"

categories = ["Normal cases","Bengin cases","Malignant cases"]

for i, category in enumerate(categories):
    path = os.path.join(dataset_path, category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path)
        image = cv2.resize(image,(128,128))

        data.append(image)
        labels.append(i)

data = np.array(data)/255.0
labels = np.array(labels)

labels = to_categorical(labels,3)

X_train,X_test,y_train,y_test = train_test_split(
    data,labels,test_size=0.2,random_state=42
)

model = Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dense(3,activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test))

model.save("lung_cancer_model.h5")