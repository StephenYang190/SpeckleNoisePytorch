from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import concatenate, BatchNormalization, Activation, Lambda, PReLU

hideChannels = 64


class OctaveConv2D(layers.Layer):
    def __init__(self, filters, alpha, kernel_size=(3, 3), strides=(1, 1),
                 padding="same", kernel_initializer="glorot_uniform",
                 kernel_regularizer=None, kernel_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        self.low_channels = int(self.filters * self.alpha)
        self.high_channels = self.filters - self.low_channels

        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        high_in = int(input_shape[0][3])
        low_in = int(input_shape[1][3])

        self.high_to_high_kernel = self.add_weight(name="high_to_high_kernel",
                                                   shape=(*self.kernel_size, high_in, self.high_channels),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint)

        self.high_to_low_kernel = self.add_weight(name="high_to_low_kernel",
                                                  shape=(*self.kernel_size, high_in, self.low_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        self.low_to_high_kernel = self.add_weight(name="low_to_high_kernel",
                                                  shape=(*self.kernel_size, low_in, self.high_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        self.low_to_low_kernel = self.add_weight(name="low_to_low_kernel",
                                                 shape=(*self.kernel_size, low_in, self.low_channels),
                                                 initializer=self.kernel_initializer,
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)
        super().build(input_shape)

    def call(self, inputs):
        high_input, low_input = inputs

        high_to_high = K.conv2d(high_input, self.high_to_high_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last"
                                )

        high_to_low = K.pool2d(high_input, (2, 2), strides=(2, 2), pool_mode="avg")
        high_to_low = K.conv2d(high_to_low, self.high_to_low_kernel,
                               strides=self.strides, padding=self.padding,
                               data_format="channels_last")

        low_to_low = K.conv2d(low_input, self.low_to_low_kernel,
                              strides=self.strides, padding=self.padding,
                              data_format="channels_last")

        low_to_high = K.conv2d(low_input, self.low_to_high_kernel,
                               strides=self.strides, padding=self.padding,
                               data_format="channels_last")
        low_to_high = K.repeat_elements(low_to_high, 2, axis=1)  # Nearest Neighbor Upsampling
        low_to_high = K.repeat_elements(low_to_high, 2, axis=2)

        high_add = high_to_high + low_to_high
        low_add = low_to_low + high_to_low

        return [high_add, low_add]

    def compute_output_shape(self, input_shapes):
        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (*high_in_shape[0:3], self.high_channels)
        low_out_shape = (*low_in_shape[0:3], self.low_channels)
        return [high_out_shape, low_out_shape]

    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "filters": self.filters,
            "alpha": self.alpha,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint
        }
        return out_config


class OctaveConv2DTranspose(layers.Layer):

    def __init__(self, filters, alpha, kernel_size=(3, 3), strides=(1, 1),
                 padding="same", kernel_initializer="glorot_uniform",
                 kernel_regularizer=None, kernel_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.filters = filters
        self.dilation_rate = (1, 1)
        # optional values
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        # -> Low Channels
        self.low_channels = int(self.filters * self.alpha)
        # -> High Channles
        self.high_channels = int(self.filters - self.low_channels)

    def build(self, input_shape):
        high_in = int(input_shape[0][3])
        low_in = int(input_shape[1][3])

        super().build(input_shape)

    def call(self, inputs):
        high_input, low_input = inputs

        high_to_high = layers.Con2DTranspose(self.high_channels, self.kernel_size,
                                             strides=self.strides, padding=self.padding,
                                             data_format="channels_last")(high_input)

        high_to_low = layers.AvgPool2D((2, 2), strides=(2, 2))(high_input)
        high_to_low = layers.Con2DTranspose(self.low_channels, self.kernel_size,
                                            strides=self.strides, padding=self.padding,
                                            data_format="channels_last")(high_to_low)

        low_to_low = layers.Con2DTranspose(self.low_channels, self.kernel_size,
                                           strides=self.strides, padding=self.padding,
                                           data_format="channels_last")(low_input)

        low_to_high = layers.Con2DTranspose(self.high_channels, self.kernel_size,
                                            strides=self.strides, padding=self.padding,
                                            data_format="channels_last")(low_input)
        low_to_high = layers.UpSampling2D((2, 2), data_format="channels_last",
                                          interpolation='nearest')(low_to_high)

        high_add = high_to_high + low_to_high
        low_add = high_to_low + low_to_low
        return [high_add, low_add]

    def compute_output_shape(self, input_shapes):
        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (*high_in_shape[:3], self.high_channels)
        low_out_shape = (*low_in_shape[:3], self.low_channels)
        return [high_out_shape, low_out_shape]

    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "filters": self.filters,
            "alpha": self.alpha,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint
        }

        return out_config


class ResBlock(layers.Layer):

    def __init__(self, dim, alpha, kernel_initializer="he_normal", **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.dim = dim
        self.Ocon1 = OctaveConv2D(self.dim, self.alpha, kernel_initializer=kernel_initializer)
        self.Ocon2 = OctaveConv2D(self.dim, self.alpha, kernel_initializer=kernel_initializer)
        self.batchnH = BatchNormalization()
        self.batchnL = BatchNormalization()
        self.batchnH1 = BatchNormalization()
        self.batchnL1 = BatchNormalization()
        self.peH = PReLU(shared_axes=[1, 2])
        self.peL = PReLU(shared_axes=[1, 2])

    def call(self, inputs):
        high_in, low_in = inputs
        x, y = inputs

        #First conv
        x, y = self.Ocon1([x, y])

        x = self.batchnH(x)
        x = self.peH(x)

        y = self.batchnL(y)
        y = self.peL(y)

        #Second conv
        x, y = self.Ocon2([x, y])
        x = self.batchnH1(x)
        y = self.batchnL1(y)

        #Addition opr
        x = x + high_in
        y = y + low_in
        return [x, y]

class ResMBlock(layers.Layer):

    def __init__(self, dim, alpha, k, kernel_initializer="he_normal", **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.dim = dim
        self.k = k
        self.Ocon1 = OctaveConv2D(self.dim, self.alpha, kernel_initializer=kernel_initializer)
        self.Ocon2 = OctaveConv2D(self.dim, self.alpha, kernel_initializer=kernel_initializer)
        self.batchnH = BatchNormalization()
        self.batchnL = BatchNormalization()
        self.batchnH1 = BatchNormalization()
        self.batchnL1 = BatchNormalization()
        self.peH = PReLU(shared_axes=[1, 2])
        self.peL = PReLU(shared_axes=[1, 2])

    def call(self, inputs):
        high_in, low_in = inputs
        x, y = inputs

        #First conv
        x, y = self.Ocon1([x, y])

        x = self.batchnH(x)
        x = self.peH(x)

        y = self.batchnL(y)
        y = self.peL(y)

        #Median layer
        x = Lambda(medianPool2D, arguments={'k': self.k},
                          output_shape=median_pool2d_output_shape)(x)
        y = Lambda(medianPool2D, arguments={'k': self.k},
                          output_shape=median_pool2d_output_shape)(y)
        #Second conv
        x, y = self.Ocon2([x, y])
        x = self.batchnH1(x)
        y = self.batchnL1(y)

        #Addition opr
        x = x + high_in
        y = y + low_in
        return [x, y]


def find_medians(input, k=3):
    ksizes = [1, k, k, 1]
    stride = [1, 1, 1, 1]
    rate = [1, 1, 1, 1]
    patches = tf.image.extract_patches(input,
                                       sizes=ksizes,
                                       strides=stride,
                                       rates=rate,
                                       padding='SAME')
    medianIndex = int(k * k / 2 - 1)
    top, _ = tf.math.top_k(patches, medianIndex, sorted=True)
    median = tf.slice(top, [0, 0, 0, medianIndex - 1], [-1, -1, -1, -1])

    return median


def medianPool2D(input, k=3):

    channels = tf.split(input, num_or_size_splits=input.shape[3], axis=3)

    channels = [find_medians(channel, k) for channel in channels]

    median = concatenate(channels, axis=-1)

    return median


def median_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)


class AdaptiveMBlock(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        high_in = input_shape[3]

        self.high_kernel = self.add_weight(name="high_kernel",
                                           shape=(1, 1, high_in * 3, high_in),
                                           initializer='glorot_uniform')

    def call(self, inputs):
        high = inputs

        medianP1 = Lambda(medianPool2D, arguments={'k': 7},
                          output_shape=median_pool2d_output_shape)(high)
        medianP2 = Lambda(medianPool2D, arguments={'k': 3},
                          output_shape=median_pool2d_output_shape)(high)
        medianP3 = Lambda(medianPool2D, arguments={'k': 5},
                          output_shape=median_pool2d_output_shape)(high)
        high_in = concatenate([medianP1, medianP2, medianP3], axis=-1)
        high_out = K.conv2d(high_in, self.high_kernel,
                            strides=(1, 1), padding="same",
                            data_format="channels_last")

        return high_out


class SNRNetWork(Model):

    def __init__(self, alpha, layerNum, outputDim=3, originialDim=3,
                 name='seNetWork', **kwargs):

        super(seNetWork, self).__init__(name=name, **kwargs)
        #Parameters
        self.originialDim = originialDim
        self.outputDim = outputDim
        self.alpha = alpha
        self.layerNum = layerNum
        #Layers
        self.octaveCon = [OctaveConv2D(hideChannels, self.alpha, kernel_initializer="Orthogonal") 
                            for i in range(self.layerNum * 2)]

        self.resCon = [ResBlock(hideChannels, self.alpha) 
                        for i in range(self.layerNum * 2)]

        self.AdaptiveMBlockConH = [AdaptiveMBlock() for i in range(self.layerNum)]
        self.AdaptiveMBlockConL = [AdaptiveMBlock() for i in range(self.layerNum)]

        self.batchnH = [BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
                        for i in range(self.layerNum * 2)]
        self.batchnL = [BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
                        for i in range(self.layerNum * 2)]

        self.proInp = OctaveConv2D(hideChannels, self.alpha, kernel_initializer="he_normal")
        self.proOut = layers.Conv2D(filters=self.outputDim, kernel_size=(3, 3),
                                     padding="same", kernel_initializer="Orthogonal")
        #ROI_guaidence
        self.proOutL = layers.Conv2D(filters=self.outputDim, kernel_size=(3, 3),
                                     padding="same", kernel_initializer="Orthogonal")
        self.proOutL = layers.Conv2D(filters=self.outputDim, kernel_size=(3, 3),
                                    padding="same", kernel_initializer="Orthogonal")

        self.peH = PReLU(shared_axes=[1, 2])
        self.peL = PReLU(shared_axes=[1, 2])

    def call(self, inputs):

        high = inputs
        low = layers.AveragePooling2D((2, 2), strides=(2, 2))(high)
        high, low = self.proInp([high, low])
        high = self.peH(high)
        low = self.peL(low)

        for i in range(self.layerNum * 2):
            high, low = self.octaveCon[i]([high, low])

            high = self.batchnH[i](high)
            high = Activation('relu')(high)

            low = self.batchnL[i](low)
            low = Activation('relu')(low)

            if i < self.layerNum:
                high = self.AdaptiveMBlockConH[i](high)
                low = self.AdaptiveMBlockConL[i](low)
                
            high, low = self.resCon[i]([high, low])

        low = layers.UpSampling2D((2, 2), data_format="channels_last",
                                  interpolation='nearest')(low)
        
        ROI_high = self.proOutH(high)
        ROI_low = self.proOutL(low)
        output = high + low

        output = self.proOut(output)
        #output = concatenate([output, output, output], axis=-1)

        return [output, ROI_high, ROI_low]
