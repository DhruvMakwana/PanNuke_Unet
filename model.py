import tensorflow as tf
from config import *

class FVT_Block(tf.keras.layers.Layer):
    def __init__(self):
        super(FVT_Block, self).__init__()
        self.n1 = tf.keras.layers.LayerNormalization()
        self.n2 = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(512, activation = 'relu')


    def call(self, x, training = True):
        inputs = x
        inputs = self.FFT(inputs)
        xt = self.n1(inputs+x)
        xd = xt
        xd = self.dense(xd)
        xi = self.n2(xd + xt)
        xi = tf.cast(xi, tf.complex64)
        return tf.math.real(tf.signal.ifft2d(xi))

    def FFT(self, x):
        xc = tf.cast(x, tf.complex64)
        x_fft = tf.math.real(tf.signal.fft2d(xc))
        return x_fft

class FANet:
    @staticmethod
    def build(width = 256, height = 256, channel = 3, classes = 6):
        inputs = tf.keras.layers.Input(shape = (height, width, channel))
        
        conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = tf.keras.layers.SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = tf.keras.layers.SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = tf.keras.layers.SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = tf.keras.layers.SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = tf.keras.layers.Dropout(0.25)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(drop4)

        conv5 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = tf.keras.layers.SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = tf.keras.layers.Dropout(0.25)(conv5)

        x = tf.keras.layers.Reshape((256, 512))(drop5)
        add1 = x
        x = FVT_Block()(x)
        #Add()([x, add1])
        for _ in range(NUM_BLOCKS - 1):
            res = x
            x = FVT_Block()(x)
            x = tf.keras.layers.Add()([res, x])
        sum = x
        re1 = tf.keras.layers.Reshape((16, 16, 512))(sum)

        up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2, 2))(re1))
        merge6 = tf.keras.layers.concatenate([drop4, up6], axis = 3)
        conv6 = tf.keras.layers.SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2, 2))(conv6))
        merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
        conv7 = tf.keras.layers.SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2, 2))(conv7))
        merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
        conv8 = tf.keras.layers.SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2, 2))(conv8))
        merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
        conv9 = tf.keras.layers.SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        
        conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = tf.keras.layers.Conv2D(classes, 1, activation = "softmax", name = 'segmentation')(conv9)

        model = tf.keras.models.Model(inputs = inputs, outputs = conv10)
        return model