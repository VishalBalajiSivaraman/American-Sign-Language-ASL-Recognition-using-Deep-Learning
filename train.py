import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import os

data = []
labels = []
classes = 24
cur_path = os.getcwd() #To get current directory


classs = { 1:"A/a",
           2:"B/b",
           3:"C/c",
           4:"D/d",
           5:"E/e",
           6:"F/f",
           7:"G/g",
           8:"H/h",
           9:"I/i",
           10:"K/k",
           11:"L/l",
           12:"M/m",
           13:"N/n",
           14:"O/o",
           15:"P/p",
           16:"Q/q",
           17:"R/r",
           18:"S/s",
           19:"T/t",
           20:"U/u",
           21:"V/v",
           22:"W/w",
           23:"X/x",
           24:"Y/y",
           }


#Retrieving the images and their labels
print("Obtaining Images & its Labels..............")
for i in range(classes):
    path = os.path.join(cur_path,'dataset/train/',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image=ImageOps.grayscale(image)
            image = image.resize((64,64))
            image = np.array(image)
            data.append(image)
            labels.append(i)
            print("{0} Loaded".format(a))
        except:
            print("Error loading image")
print("Dataset Loaded")

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

Img=64
#Converting the labels into one hot encoding
y_train = tf.keras.utils.to_categorical(y_train,24)
y_test = tf.keras.utils.to_categorical(y_test,24)
X_train=np.array(X_train).reshape(-1,Img,Img,1)
X_test=np.array(X_test).reshape(-1,Img,Img,1)
print("Training under process...")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(24, activation='softmax'))
print("Initialized model")
# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train, y_train, batch_size=15, epochs=15, validation_data=(X_test, y_test))
model.save("ASL.h5")

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('Accuracy.png')

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('Loss.png')
print("Saved Model & Graph to disk")
