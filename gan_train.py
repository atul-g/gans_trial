import os
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


#making losses:
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_ouput), real_output)
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    total_loss=real_loss+fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


###### saving checkpoints

checkpoint_dir = './training_checkpoints'
checkpoint_prefix=os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=gen_model,
                                 discriminator=discr_model)

noise=tf.random.normal([1, 100])
EPOCHS=50
noise_dim=100
num_examples_to_generate=16

seed=tf.random.normal(['num_examples_to_generate', noise_dim])

def train_step(images):
    noise=tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape"
        generated_images = gen_model(noise, training=True)
        
        real_output=discr_model(images, training=True)
        fake_output=discr_model(generated_images, training=True)
        
        discriminator_loss=discriminator_loss(real_output, fake_output)
        gen_loss=generator_loss(fake_output)
        
    gradients_of_generator=gen_tape.gradient(gen_loss, gen_model.trainbable, variables)
    disc_loss=discriminator_loss(real_output, fake_output)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discr_model.trainable_variables)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discr_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discr_model.trainable_variables))



