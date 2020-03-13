import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm

cwd=os.getcwd()
BATCH_SIZE=32
TOTAL_IMAGES=16661
NO_OF_BATCHES=(TOTAL_IMAGES//BATCH_SIZE)+1 if(TOTAL_IMAGES%BATCH_SIZE) else TOTAL_IMAGES//BATCH_SIZE

image_gen=ImageDataGenerator(rescale=1./255)
train_gen=image_gen.flow_from_directory(batch_size=32, directory='dataset', shuffle='True', target_size=(112,112), class_mode=None, color_mode='grayscale')

j=1

with tqdm(total=NO_OF_BATCHES, desc=f"EPOPCH {j}") as t:
	for x in train_gen: #x is the batch tensor. NOTE train_gen loops infinitely, you have to explicitly break over it.
	   # print(x)
		#print(type(x))
		#print(x.shape[0])
		y=x[0]
		y=y.reshape(112,112)
		#print(y.shape)
		plt.imshow(y, cmap='gray')
		plt.show()
		#print('IMAGE SHOWED')
		t.update()
