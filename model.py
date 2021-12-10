#imports
import os
from PIL.Image import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance
import time

#keras imports
from keras.models import load_model as load
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

#print for progress check
print('imports completed...')

#using Haar-cascade classifier for object detection. Specifically, for getting the boundary box around a person's face -> face detection

#base directory for opencv
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
#frontal face default xml
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

#import cascade model
face_model = cv2.CascadeClassifier(haar_model)

print('cascade model loaded in...')



#Sample image test (commented out after first run is successful)
img = cv2.imread('unnamed.jpeg')
print('image read in successfully...')
#grayscale
img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

#returns a list of (x,y,w,h) tuples
faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4)

#social distance violations
MIN_DISTANCE = 130

#train-test-val data split
train_dir = "faceMaskData_12k/Train"
test_dir = "faceMaskData_12k/Test"
val_dir = "faceMaskData_12k/Validation"

#data augmentation; generates batches of tensor image data with real-time augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2)
train_generator= train_datagen.flow_from_directory(directory=train_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

print('data augmentation complete...')

#model
vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(128,128,3))

# for layer in vgg19.layers:
#     layer.trainable = False
    
# model = Sequential()
# model.add(vgg19)
# model.add(Flatten())
# model.add(Dense(2,activation='sigmoid'))
# model.summary()

# #compile model
# model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")

# print('model compiled successfully...')

# #train model
# history = model.fit_generator(generator=train_generator,
#                               steps_per_epoch=len(train_generator)//32,
#                               epochs=20,validation_data=val_generator,
#                               validation_steps=len(val_generator)//32)

# #evaluate model
# scores = model.evaluate(test_generator)
# #print off evaluation results
# print(scores)

# #save model
# model.save('masknet.h5')

# print('model saved...')

#load model
model = load('masknet.h5')
print('model loaded...')

#label data
mask_label = {0:'MASK',1:'NO MASK'}
dist_label = {0:(0,255,0),1:(255,0,0)}

#generate plot and graphic
print(faces)
if len(faces)>=2:
    label = [0 for i in range(len(faces))]
    for i in range(len(faces)-1):
        for j in range(i+1, len(faces)):
            dist = distance.euclidean(faces[i][:2],faces[j][:2])
            if dist<MIN_DISTANCE:
                label[i] = 1
                label[j] = 1
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        crop = new_img[y:y+h,x:x+w]
        crop = cv2.resize(crop,(128,128))
        crop = np.reshape(crop,[1,128,128,3])/255.0
        mask_result = model.predict(crop)
        cv2.putText(new_img,mask_label[mask_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,1,2)
        cv2.rectangle(new_img,(x,y),(x+w,y+h),dist_label[label[i]],1)
    plt.figure(figsize=(10,10))
    plt.imshow(new_img)
    plt.show()
else:
    print('only one face recognized')
