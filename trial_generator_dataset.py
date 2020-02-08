import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

cwd=os.getcwd()
image_gen=ImageDataGenerator(rescale=1./255)
train_gen=image_gen.flow_from_directory(batch_size=5, directory='dataset', shuffle='True', target_size=(28,28), class_mode=None)

j=1
for x in train_gen: #x is the batch tensor. NOTE train_gen loops infinitely, you have to explicitly break over it.
   # print(x)
    #print(type(x))
    print(x.shape[0])
