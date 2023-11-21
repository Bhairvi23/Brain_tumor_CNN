# Brain_tumor_CNN
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import tensorflow as tf
from tensorflow import keras
from keras import Sequential,models,layers
from tensorflow.keras.preprocessing import image
from numpy import asarray
import PIL
from PIL import Image
import os
import cv2
from sklearn.model_selection import train_test_split


#Importing Datasets
img_dir='../input/brain-mri-images-for-brain-tumor-detection/'
no_images=os.listdir(img_dir + 'no/')
yes_images=os.listdir(img_dir + 'yes/')
dataset=[]
lab=[]


#No tumour images
for image_name in no_images:
    image=cv2.imread(img_dir + 'no/' +image_name)
    image=Image.fromarray(image,'RGB')
    image=image.resize((112,112))
    dataset.append(np.array(image))
    lab.append(0)

#Tumour images
for image_name in yes_images:
    image=cv2.imread(img_dir + 'yes/' +image_name)
    image=Image.fromarray(image,'RGB')
    image=image.resize((112,112))
    dataset.append(np.array(image))
    lab.append(1)

data=np.asarray(dataset)
l=np.asarray(lab)
print(data.shape, l.shape)


print('Total Number of Image: ',len(l))

#Sample Images
plt.imshow(data[2])


x_train,x_test,y_train,y_test = train_test_split(data,l, test_size=0.3, shuffle=True, random_state=0)

model=Sequential([
                     
                     #cnn
                    layers.Conv2D(32,(3,3),activation="relu",input_shape=(110,110,3)),
                    layers.MaxPooling2D((2,2)),
     
                    layers.Conv2D(64,(3,3),activation="relu"),
                    layers.MaxPooling2D((2,2)),
    
                    layers.Conv2D(128,(3,3),activation="relu"),
                    layers.MaxPooling2D((2,2)),
                    
                    layers.Conv2D(128,(3,3),activation="relu"),
                    layers.MaxPooling2D((2,2)),
    
                     #dense_layer
                     layers.Flatten(),
                     layers.Dense(512,activation="relu"),
                     layers.Dense(2,activation="softmax")


])

print(model.summary())


model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs = 100, batch_size = 100, verbose = 1,validation_data = (x_test, y_test))

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=16)
print("test loss, test acc:", results)
