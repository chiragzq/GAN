from tensorflow.keras.layers import Conv2D, Dropout, LeakyReLU, Flatten, Dense, Activation, BatchNormalization, Reshape, UpSampling2D, Conv2DTranspose
import tensorflow.keras as keras
import tensorflow as tf

class ModelGenerator:
    def __init__(self):
        self.d = None
        self.g = None
        
    def discriminator(self):
        if self.d == None:
            input_shape = (28, 28, 1)
            depth = 64
            dropout = 0.4
            kernel_size = 5
            strides = 2
            self.d = keras.models.Sequential([
                Conv2D(
                    depth,
                    kernel_size,
                    strides=strides,
                    input_shape=input_shape,
                    padding="same",
                    activation=LeakyReLU(alpha=0.2)),
                Dropout(dropout),
                Conv2D(
                    depth*2,
                    kernel_size,
                    strides=strides,
                    padding="same",
                    activation = LeakyReLU(alpha=0.2)),
                Dropout(dropout),
                Conv2D(
                    depth*4,
                    kernel_size,
                    strides=strides,
                    padding="same",
                    activation = LeakyReLU(alpha=0.2)),
                Dropout(dropout),
                Conv2D(
                    depth*8,
                    kernel_size,
                    strides=strides,
                    padding="same",
                    activation = LeakyReLU(alpha=0.2)),
                Dropout(dropout),
                Flatten(),
                Dense(1),
                Activation("sigmoid")
            ])
        return self.d
    
    def generator(self):
        if self.g == None:
            dropout = 0.4
            depth = 64 * 4
            dim = int(28 / 4)
            input_dim = 100
            momentum = 0.9
            kernel_size = 5
            self.g = keras.models.Sequential([
                Dense(dim * dim * depth, input_dim=input_dim),
                BatchNormalization(momentum=momentum),
                Activation("relu"),
                Reshape((dim, dim, depth)),
                Dropout(dropout),
                UpSampling2D(),
                Conv2DTranspose(int(depth // 2), kernel_size, padding="same"),
                BatchNormalization(momentum=momentum),
                Activation("relu"),
                UpSampling2D(),
                Conv2DTranspose(int(depth // 4), kernel_size, padding="same"),
                BatchNormalization(momentum=momentum),
                Activation("relu"),
                Conv2DTranspose(int(depth // 8), kernel_size, padding="same"),
                BatchNormalization(momentum=momentum),
                Activation("relu"),
                Conv2DTranspose(1, kernel_size, padding="same"),
                Activation("sigmoid"),
            ])
        return self.g