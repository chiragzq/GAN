import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop
from model_generator import ModelGenerator
import server
print()
print()

print("Starting server")
server.start_server()

model_generator = ModelGenerator()

discriminator_optimizer = RMSprop(lr = 0.0008, clipvalue = 1.0, decay=6e-8)
discriminator = keras.models.Sequential([model_generator.discriminator()])
discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer, metrics=["accuracy"])
print("Compiled discriminator")

adversial_optimizer = RMSprop(lr = 0.0004, clipvalue=1.0, decay=3e-8)
adversial = keras.models.Sequential([
    model_generator.generator(),
    model_generator.discriminator()
])
adversial.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer, metrics=["accuracy"])
print("Compiled adversial")

print("Loading images")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Done")


#batch_size = 256
#train_images = np.array([x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :]])
#noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
#fake_train_images = model_generator.generator().predict(noise)

#print(train_images.shape)
#print(fake_train_images.shape)
#samples = np.concatenate((train_images, fake_train_images))
#labels = np.ones([2 * batch_size, 1])
#labels[batchsize:, :] = 0
#discriminator_loss = discriminator.train_on_batch(samples, labels)


#labels = np.ones([batch_size, 1])
#noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
#adversial_loss = adversial.train_on_batch(noise, labels)
