import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import ImageOps
from tensorflow.keras.preprocessing import image# used for preproccesing 
model = load_model('ASL.h5')
print("Loaded model from disk")

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

Img=64
def classify(img_file):
    test_image=image.load_img(img_file)
    test_image=ImageOps.grayscale(test_image)
    test_image = test_image.resize((64, 64))
    test_image = np.expand_dims(test_image, axis=0)
    test = np.array(test_image).reshape(-1,Img,Img,1)
    result = model.predict_classes(test)[0]
    sign = classs[result + 1]
    print("The character is ",sign)
    
print("Obtaining Images & its Labels..............")
path='D:/python/dl programs/American Sign Language/dataset/test'
files=[]
print("Dataset Loaded")
# r=root,d=directories,f=files
for r,d,f in os.walk(path):
    for file in f:
        if '.jpeg' or '.jpg' or '.png' or '.JPG' in file:
            files.append(os.path.join(r,file))
for f in files:
    classify(f)
    print('\n')
