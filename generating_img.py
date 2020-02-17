#import time
#import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
#from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Conv2DTranspose, Reshape, Conv2D, Flatten, Dropout
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

'''

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
                        

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=gen_model,
                                 discriminator=discr_model)
                                 

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


#GENERATING AN IMAGE

noise = tf.random.normal([1, 100])
generated_image = gen_model(noise, training=False)
print(f"\n\n{generated_image}\n\n{generated_image.shape}")
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()




'''

#USING THE SAVED gen_model_50.h5 TO GENERATE IMAGES:


gen_model=load_model("gen_model_50.h5")

noise=tf.random.normal([1,100])
generated_image=gen_model(noise, training=False)
plt.imshow(generated_image[0,:,:,0], cmap='gray')
plt.show()





