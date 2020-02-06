import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Conv2DTranspose, Reshape, Conv2D, Flatten, Dropout

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

gen_model.summary()

generated_image=gen_model(tf.random.normal([1,100]))
#print(generated_image)

discr_model=Sequential([
                        Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]),
                        LeakyReLU(),
                        Dropout(0.3),
                        Conv2D(128, (5,5), strides=(2,2), padding='same'),
                        LeakyReLU(),
                        Dropout(0.3),
                        Flatten(),
                        Dense(1)
                        ])
discr_model.summary()
