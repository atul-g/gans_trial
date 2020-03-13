import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Conv2DTranspose, Reshape, Conv2D, Flatten, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

'''
########## ORIGINAL GENERATOR SHAPE #################

gen_model=Sequential([
                    Dense(7*7*256, use_bias=False, input_shape=(100,)),
                    BatchNormalization(),
                    LeakyReLU(),
                    
                    Reshape((7,7,256)),
                    Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False),
                    BatchNormalization(),
                    LeakyReLU(),
                    Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
                    BatchNormalization(),
                    LeakyReLU(),
                    Conv2DTranspose(1,(5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
                    ])

print(gen_model.summary())
'''

########### ALTERNATE MODELS #################
'''
#model1:
gen_model=Sequential([
                    Dense(7*7*256, use_bias=False, input_shape=(100,)),
                    BatchNormalization(),
                    LeakyReLU(),
                    
                    Reshape((7,7,256)),
                    Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
                    BatchNormalization(),
                    LeakyReLU(),
                    Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
                    BatchNormalization(),
                    LeakyReLU(),
                    Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
                    BatchNormalization(),
                    LeakyReLU(),
                    Conv2DTranspose(1,(4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh')
                    ])

gen_model.summary()
'''

#model 2
gen_model=Sequential([
                    Dense(128*16*16, input_shape=(100,)),
                    BatchNormalization(momentum=0.9),
                    LeakyReLU(alpha=0.1),
                    
                    Reshape((16,16,128)),
                    Conv2D(128, (5,5), strides=(1,1), padding='same'),
                    BatchNormalization(momentum=0.9),
                    LeakyReLU(alpha=0.1),
                    
                    Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False),
                    BatchNormalization(momentum=0.9),
                    LeakyReLU(alpha=0.1),
                    
                    Conv2D(128, (5,5), strides=(1,1), padding='same'),
                    BatchNormalization(momentum=0.9),
                    LeakyReLU(alpha=0.1),
                    
                    Conv2D(128, (5,5), strides=(1,1), padding='same'),
                    BatchNormalization(momentum=0.9),
                    LeakyReLU(alpha=0.1),
                    
                    Conv2D(3, (5,5), strides=(1,1), padding='same'),
                    BatchNormalization(momentum=0.9),
                    LeakyReLU(alpha=0.1),
                    
                    Activation("tanh")
                    ])

print(gen_model.summary())



