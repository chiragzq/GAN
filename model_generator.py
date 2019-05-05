from keras.layers import Input, Conv2D, Dropout, LeakyReLU, Flatten, Dense, Activation, BatchNormalization, Reshape, UpSampling2D, Conv2DTranspose
from keras.models import Model
import keras
import tensorflow as tf

class ModelGenerator:
    def __init__(self):
        self.d = None
        self.g = None
        
    def discriminator(self):
        if self.d == None:
            input_shape = (28, 28, 1)
            kernel_size = 5
            strides = 2

            filters = [32, 64, 128, 256]
            strides = [2, 2, 2, 1]

            input = Input(input_shape)
            model = input
            for filter_size, stride in zip(filters, strides):
                model = LeakyReLU(alpha=0.2)(model)
                model = Conv2D(filters=filter_size, kernel_size=kernel_size, strides=stride, padding="same")(model)
            model = Flatten()(model)
            model = Dense(1)(model)
            model = Activation("sigmoid")(model)
            self.d = Model(input, model)
            #self.d = keras.models.Sequential([
            #    Conv2D(
            #        depth,
            #        kernel_size,
            #        strides=strides,
            #        input_shape=input_shape,
            #        padding="same",
            #        activation=LeakyReLU(alpha=0.2)),
            #    Dropout(dropout),
            #    Conv2D(
            #        depth*2,
            #        kernel_size,
            #        strides=strides,
            #        padding="same",
            #        activation = LeakyReLU(alpha=0.2)),
            #    Dropout(dropout),
            #    Conv2D(
            #        depth*4,
            #        kernel_size,
            #        strides=strides,
            #        padding="same",
            #        activation = LeakyReLU(alpha=0.2)),
            #    Dropout(dropout),
            #    Conv2D(
            #        depth*8,
            #        kernel_size,
            #        strides=strides,
            #        padding="same",
            #        activation = LeakyReLU(alpha=0.2)),
            #    Dropout(dropout),
            #    Flatten(),
            #    Dense(1),
            #    Activation("sigmoid")
            #])
            #self.d.summary()
        return self.d
    
    def generator(self):
        if self.g == None:
            dropout = 0.4
            dim = int(28 / 4)
            input_dim = 100
            momentum = 0.9
            kernel_size = 5
            z_vector = 100

            filters = [128, 64, 32, 1]
            strides = [2, 2, 1, 1]

            input = Input(shape=(z_vector,))
            model = Dense(dim * dim * filters[0], input_dim=100)(input)
            model = Reshape((dim, dim, filters[0]))(model)

            for layer_filter, stride in zip(filters, strides):
                model = BatchNormalization(momentum=momentum)(model)
                model = Activation("relu")(model)
                model = Conv2DTranspose(filters=layer_filter, kernel_size=kernel_size, strides=stride, padding="same")(model)
            model = Activation("sigmoid")(model)
            self.g = Model(input, model)
            #self.g = keras.models.Sequential([
            #    Dense(dim * dim * depth, input_dim=input_dim),
            #    BatchNormalization(momentum=momentum),
            #    Activation("relu"),
            #    Reshape((dim, dim, depth)),
            #    Dropout(dropout),
            #    UpSampling2D(),
            #    Conv2DTranspose(int(depth // 2), kernel_size, padding="same"),
            #    BatchNormalization(momentum=momentum),
            #    Activation("relu"),
            #    UpSampling2D(),
            #    Conv2DTranspose(int(depth // 4), kernel_size, padding="same"),
            #    BatchNormalization(momentum=momentum),
            #    Activation("relu"),
            #    Conv2DTranspose(int(depth // 8), kernel_size, padding="same"),
            #    BatchNormalization(momentum=momentum),
            #    Activation("relu"),
            #    Conv2DTranspose(1, kernel_size, padding="same"),
            #    Activation("sigmoid"),
            #])
            #self.g.summary()
        return self.g