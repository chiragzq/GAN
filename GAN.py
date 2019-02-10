import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop

import numpy as np

from model_generator import ModelGenerator

graph = tf.get_default_graph()

class GAN:
    def __init__(self):
        self.model_generator = ModelGenerator()
        self.discriminator = None
        self.adversial = None
        self.generator = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def initialize_models(self):
        discriminator_optimizer = RMSprop(lr = 0.0008, clipvalue = 1.0, decay=6e-8)
        self.discriminator = keras.models.Sequential([self.model_generator.discriminator()])
        self.discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer, metrics=["accuracy"])
        print("Compiled discriminator")

        self.model_generator.discriminator().traininable = False
        adversial_optimizer = RMSprop(lr = 0.0004, clipvalue=1.0, decay=3e-8)
        self.adversial = keras.models.Sequential([
            self.model_generator.generator(),
            self.model_generator.discriminator()
        ])
        self.adversial.compile(loss="binary_crossentropy", optimizer=adversial_optimizer, metrics=["accuracy"])
        print("Compiled adversial")
    
    def load_images(self):
        print("Loading images")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        print("Done!")
    
    def train(self, iterations=10, batch_size=256):
        print("Training for %d iterations" % iterations)
        with graph.as_default():
            for i in range(0, iterations):
                train_images = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, None]
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                fake_train_images = self.model_generator.generator().predict(noise)[:, :, :]
                samples = np.concatenate((train_images, fake_train_images))
                labels = np.ones([2 * batch_size, 1])
                labels[batch_size:, :] = 0
                discriminator_loss = self.discriminator.train_on_batch(samples, labels)
                labels = np.ones([batch_size, 1])
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                adversial_loss = self.adversial.train_on_batch(noise, labels)
                #adversial_loss = (0, 0)
                print("\rTraining progress: %d/%d | Discriminator loss: %f, accuracy: %f | Adversial loss: %f, accuracy: %f" %
                    (i + 1, iterations, discriminator_loss[0], discriminator_loss[1], adversial_loss[0], adversial_loss[1]), end="")
        print("\nDone!")

    def generate_image(self):
        global graph
        with graph.as_default():
            noise = np.random.uniform(-1.0, 1.0, size=[1, 100])
            return self.model_generator.generator().predict(noise)

    def predict_image(self, image):
        global graph
        with graph.as_default():
            return self.discriminator.predict(image)
        #Check if an image is a number using the discriminator
