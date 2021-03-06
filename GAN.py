import tensorflow as tf
#import tensorflow.keras as keras
import keras
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.layers import Input
from keras.models import Model

import numpy as np
import png

from model_generator import ModelGenerator

graph = tf.get_default_graph()

do_saving = False

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
        self.training_iterations = 0
        self.saved_images = 0
        self.grid_length = 6
        self.noise = np.random.uniform(-1.0, 1.0, size=(self.grid_length ** 2, 100))


    def initialize_models(self):
        discriminator_optimizer = RMSprop(lr=2e-4, decay=6e-8)
        self.discriminator = keras.models.Sequential([self.model_generator.discriminator()])
        self.discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer, metrics=["accuracy"])
        print("Compiled discriminator")
        
        self.model_generator.discriminator().trainable = False
        for layer in self.model_generator.discriminator().layers:
            layer.trainable = False
        adversial_optimizer = RMSprop(lr=1e-4, decay=3e-8)

        input = Input(shape=(100,))
        self.adversial = Model(
            input,
            self.model_generator.discriminator()(self.model_generator.generator()(input))
        )
        self.adversial.compile(loss="binary_crossentropy", optimizer=adversial_optimizer, metrics=["accuracy"])
        #self.adversial.summary()
        
        self.generator = self.model_generator.generator()
        print("Compiled adversial")
    
    def load_images(self, number=None):
        print("Loading images")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255
        
        if number is not None:
            new_x_train = []
            for image, label in zip(self.x_train, self.y_train):
                if label == number:
                    new_x_train.append(image)
            self.y_train = np.ones([len(new_x_train)])
            self.x_train = np.array(new_x_train)
        
        #new_x_train = []
        #for _ in range(0, 512):
        #    new_x_train.append(self.x_train[0])
        #self.y_train = np.ones([len(new_x_train)])
        #self.x_train = np.array(new_x_train)
        
        print("Done!")
    
    def train(self, iterations=10, batch_size=256):
        print("Training for %d iterations" % iterations)
        with graph.as_default():
            for i in range(0, iterations):
                #train_images = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, None]
                #noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                #fake_train_images = self.model_generator.generator().predict(noise)[:, :, :]
                #samples = np.concatenate((train_images, fake_train_images))
                #labels = np.ones([2 * batch_size, 1])
                #labels[batch_size:, :] = 0
                #discriminator_loss = self.discriminator.train_on_batch(samples, labels)
                #labels = np.ones([batch_size, 1])
                #noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                #adversial_loss = self.adversial.train_on_batch(noise, labels)
                #adversial_loss = (0, 0)
                #print("\rTraining progress: %d/%d | Discriminator loss: %f, accuracy: %f | Adversial loss: %f, accuracy: %f" %
                #    (i + 1, iterations, discriminator_loss[0], discriminator_loss[1], adversial_loss[0], adversial_loss[1]), end="")
                
                rand_indexes = np.random.randint(0, self.x_train.shape[0], size=batch_size)
                real_images = self.x_train[rand_indexes, :, :, None]
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                fake_images = self.model_generator.generator().predict(noise)
                x = np.concatenate((real_images, fake_images))
                y = np.ones([2 * batch_size, 1])
                y[batch_size:, :] = 0.0
                loss, acc = self.discriminator.train_on_batch(x, y)
                log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                y = np.ones([batch_size, 1])
                loss, acc = self.adversial.train_on_batch(noise, y)
                log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
                print(log)
        self.training_iterations += iterations
        if do_saving and self.training_iterations // 50 > self.saved_images:
            self.saved_images += 1
            images = np.array(self.model_generator.generator().predict(self.noise))
            images = [np.pad(img, 1, "constant") for img in (255 - np.array(self.model_generator.generator().predict(self.noise))[:, :, :, 0] * 255).astype(np.int16)]
            edge = len(images[0])
            big = []
            
            for i in range(0, self.grid_length):
                for k in range(0, edge):
                    big.append([])
                    for j in range(0, self.grid_length):
                        for l in range(0, edge):
                            big[i * edge + k].append(images[i * self.grid_length + j][k][l])
            for i in range(0, len(big)):
                big[i] = np.array(big[i])
            big = np.array(big)
            
            png.from_array(self.stretch(big), "L").save("static/images/%05d.png" % self.training_iterations)
        
        print("\nDone!")
    
    def stretch(self, img):
        stretch_factor = 8
        res = []
        for i in range(0, len(img) * stretch_factor):
            res.append([])
            for j in range(0, len(img[0]) * stretch_factor):
                res[i].append(img[i // stretch_factor][j // stretch_factor])
        return res

    def do_discriminator_test(self):
        batch_size = 64
        rand_indexes = np.random.randint(0, 10000, size=batch_size)
        real_images = self.x_train[rand_indexes, :, :, None]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        fake_images = self.model_generator.generator().predict(noise)
        x = np.concatenate((real_images, fake_images))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0.0
        loss, acc = self.discriminator.test_on_batch(x, y)
        print("%f %f" % (loss, acc))

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        y = np.ones([batch_size, 1])
        loss, acc = self.adversial.train_on_batch(noise, y)

        x = np.concatenate((real_images, fake_images))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0.0
        loss, acc = self.discriminator.test_on_batch(x, y)
        print("%f %f" % (loss, acc))
        
    def do_adversarial_test(self):
        batch_size = 64
        rand_indexes = np.random.randint(0, 10000, size=batch_size)
        real_images = self.x_train[rand_indexes, :, :, None]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        fake_images = self.model_generator.generator().predict(noise)
        
        y = np.ones([batch_size, 1])
        loss, acc = self.adversial.test_on_batch(noise, y)
        print("%f %f" % (loss, acc))
        
        x = np.concatenate((real_images, fake_images))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0.0
        loss, acc = self.discriminator.train_on_batch(x, y)
        
        y = np.ones([batch_size, 1])
        loss, acc = self.adversial.test_on_batch(noise, y)
        
        print("%f %f" % (loss, acc))
        
                
    def generate_image(self):
        global graph
        with graph.as_default():
            noise = np.random.uniform(-1.0, 1.0, size=[1, 100])
            return self.model_generator.generator().predict(noise)

    def predict_image(self, image):
        global graph
        with graph.as_default():
            image = image[:, :, None]
            return self.discriminator.predict(np.array([image]))
        #Check if an image is a number using the discriminator
