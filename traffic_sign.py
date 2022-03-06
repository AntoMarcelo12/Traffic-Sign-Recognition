# Step 1: Explore the dataset
# 
# Our ‘train’ folder contains 43 folders each representing a different class. 
# The range of the folder is from 0 to 42. With the help of the OS module, 
# we iterate over all the classes and append images and their respective labels in the data and labels list.
# 
# The PIL library is used to open image content into an array.

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 43
#cur_path = os.getcwd()
cur_path = "C:\\Users\\Antonio\\Desktop\\Master\\python\\python-Traffic-sign-classification\\"

#Retrieving the images and their labels 
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            #sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)



# Step 2: Build a CNN model
#Building the model
# To classify the images into their respective categories, we will build a CNN model (Convolutional Neural Network). CNN is best for image classification purposes.
# 
# The architecture of our model is:
# 
# 2 Conv2D layer (filter=32, kernel_size=(5,5), activation=”relu”)
# MaxPool2D layer ( pool_size=(2,2))
# Dropout layer (rate=0.25)
# 2 Conv2D layer (filter=64, kernel_size=(3,3), activation=”relu”)
# MaxPool2D layer ( pool_size=(2,2))
# Dropout layer (rate=0.25)
# Flatten layer to squeeze the layers into 1 dimension
# Dense Fully connected layer (256 nodes, activation=”relu”)
# Dropout layer (rate=0.5)
# Dense layer (43 nodes, activation=”softmax”)
# We compile the model with Adam optimizer which performs well and loss is “categorical_crossentropy” because we have multiple classes to categorise.

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Steps 3: Train and validate the model
# After building the model architecture, we then train the model using model.fit(). 
# I tried with batch size 32 and 64. Our model performed better with 64 batch size. And after 15 epochs the accuracy was stable.

epochs = 15
# epochs = 2
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")


# #mio codice
# from keras.models import load_model
# model = load_model('my_model.h5')
# #fine mio codice

# Our model got a 95% accuracy on the training dataset. With matplotlib, we plot the graph for accuracy and the loss.
#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# Step 4: Test our model with test dataset
# 
# Our dataset contains a test folder and in a test.csv file, we have the details related to the image path and their respective class labels. 
# We extract the image path and labels using pandas. 
# Then to predict the model, we have to resize our images to 30×30 pixels and make a numpy array containing all image data. 
# From the sklearn.metrics, we imported the accuracy_score and observed how our model predicted the actual labels. 
# We achieved a 95% accuracy in this model.

#testing accuracy on test dataset
from sklearn.metrics import accuracy_score

y_test = pd.read_csv("C:\\Users\\Antonio\\Desktop\\Master\\python\\python-Traffic-sign-classification\\Test.csv")

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

for img in imgs:
    image = Image.open('C:\\Users\\Antonio\\Desktop\\Master\\python\\python-Traffic-sign-classification\\'+img)
    # image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    #data.append(image)

X_test=np.array(data)

# pred = model.predict_classes(X_test)
pred = model.predict(X_test)
# pred = model.predict([X_test])[0]
# pred=pred.astype(np.int64)


# # indice[i]=np.where(pred[i]==1)[0][0]
# massimo
indice=[]

for i in range(12630):
    # print(np.where(pred[i]==max(pred[i])))
    temp=(np.where(pred[i]==max(pred[i])))
    indice.append(temp[0][0])
    # indice(i)=temp[0][0]
    
traffic_index=np.array(indice)

#Accuracy with the test data
from sklearn.metrics import accuracy_score
# print(accuracy_score(labels, pred))
print(accuracy_score(labels, traffic_index))

model.save("traffic_classifier.h5")

