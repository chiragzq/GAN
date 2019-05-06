from GAN import GAN
from keras.models import load_model
import numpy as np
import tensorflow as tf

graph = tf.get_default_graph()

class NumberGenerator:
    def __init__(self):
        self.generators = []
        for i in range(0, 10):
            self.generators.append(load_model("models/generator" + str(i)))
    
    def generate_digit(self, digit):
        generator = self.generators[digit]
        
        global graph
        with graph.as_default():
            noise = np.random.uniform(-1.0, 1.0, size=[1, 100])
            return generator.predict(noise)[0, :, :, 0]
    
    def generate_number(self, digits):
        res = []
        for i in range(0, 28):
            res.append([])
        for digit in digits:
            image = self.reduce_image(self.generate_digit(digit))
            print(image.shape)
            for (index, r) in enumerate(image):
                res[index].extend(r)
        return np.array(res)
        
    def reduce_image(self, image):
        THRESH = 0.5
        for i in range(0, len(image)):
            flag = False
            for j in range(0, len(image)):
                if image[j][i] > THRESH:
                    flag = True
            if flag:
                break
        for j in range(len(image) - 1, -1):
            flag = False
            for k in range(0, len(image[i])):
                if image[k][j] > THRESH:
                    flag = True
            if flag:
                break
        return image[:, slice(i, j + 1, 1)]
        