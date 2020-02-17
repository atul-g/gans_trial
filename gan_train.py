import time
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Conv2DTranspose, Reshape, Conv2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

############## PREPARING DATASET
BATCH_SIZE=32
TOTAL_IMAGES=5357
NO_OF_BATCHES=(TOTAL_IMAGES//BATCH_SIZE)+1 if(TOTAL_IMAGES%BATCH_SIZE) else TOTAL_IMAGES//BATCH_SIZE

image_gen=ImageDataGenerator(rescale=1./255)
train_gen=image_gen.flow_from_directory(batch_size=BATCH_SIZE, directory='dataset', shuffle='True', target_size=(28,28),color_mode='grayscale', class_mode=None)


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
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
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

seed=tf.random.normal([num_examples_to_generate, noise_dim])

def train_step(images):
    noise=tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen_model(noise, training=True)
        
        real_output=discr_model(images, training=True)
        fake_output=discr_model(generated_images, training=True)
        
        disc_loss=discriminator_loss(real_output, fake_output)
        gen_loss=generator_loss(fake_output)
        
    
    gradients_of_generator = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discr_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discr_model.trainable_variables))
    #return (disc_loss,gen_loss) # I AM ADDING THIS LINE SO THAT I CAN PRINT OUT THE LOSSES EVERY EPOCH
#BUT I AM REALLY NOT SURE IF THESE LOSSES RETURNED ARE THE "EPOCH" LOSSES, THIS IS JUST SOME KIND OF INDICATION. NO NEED TO USE THEM FOR ACTUAL MEASURES.


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    
    i=1
    for image_batch in dataset:
      if(i>NO_OF_BATCHES):
        #train_step(image_batch)
        break
      else:
        #disc_loss,gen_loss=train_step(image_batch)
        train_step(image_batch)
        i+=1
    

    # Produce images for the GIF as we go
   #display.clear_output(wait=True)
    #generate_and_save_images(gen_model, epoch + 1, seed)
                             
    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    #print(f"Disriminator Loss: {disc_loss}, Generator Loss:{gen_loss}")


  #saving model in keras hdf5 format
  gen_model.save(os.getcwd()+f"/gen_model_{EPOCHS}.h5")
  
  # Generate after the final epoch
 #display.clear_output(wait=True)
 
  generate_and_save_images(gen_model,
                           epochs,
                           seed)



def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()




####
'''
Now this train_dataset below needs to be a tensor like:

tf.Tensor(
[[2 3]
 [5 6]], shape=(2, 2), dtype=int32)
tf.Tensor(
[[6 7]
 [3 4]], shape=(2, 2), dtype=int32)
tf.Tensor([[1 2]], shape=(1, 2), dtype=int32)


You see? it is a tensor with batch size 2 and a buffer size/total size of 5

similiarly make tesors of images with certain batch sizes
'''
train(train_gen, EPOCHS)

