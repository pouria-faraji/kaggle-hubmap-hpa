import tensorflow as tf
# from tensorflow.python.keras import Model, layers, Sequential, Input, regularizers
from tensorflow.keras import Model, layers, Sequential, Input, regularizers


class ConvolutionBlock(Model):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size = 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation(tf.nn.relu)
        # self.dropout1 = layers.Dropout(0.2)

        self.conv2 = layers.Conv2D(filters, kernel_size = 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation(tf.nn.relu)
        # self.dropout2 = layers.Dropout(0.2)        
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        # x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # x = self.dropout2(x)

        return x

class EncoderBlock(Model):
    def __init__(self, filters):
        super().__init__()
        self.cb = ConvolutionBlock(filters)
        self.pool = layers.MaxPooling2D(pool_size=(2, 2))
    
    def call(self, inputs):
        x = self.cb(inputs)
        p = self.pool(x)
        return x, p

class DecoderBlock(Model):
    def __init__(self, filters):
        super().__init__()
        self.ct = layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')
        # self.up = layers.UpSampling2D(size=(2, 2))
        self.cb = ConvolutionBlock(filters)
    
    def call(self, inputs, p):
        x = self.ct(inputs)
        x = tf.concat([x, p], axis=-1)
        x = self.cb(x)
        return x

class UNet(Model):
    def __init__(self, num_classes=2, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height

        self.encoder1 = EncoderBlock(64)
        self.encoder2 = EncoderBlock(128)
        self.encoder3 = EncoderBlock(256)
        self.encoder4 = EncoderBlock(512)

        self.cb = ConvolutionBlock(1024)

        self.decoder1 = DecoderBlock(512)
        self.decoder2 = DecoderBlock(256)
        self.decoder3 = DecoderBlock(128)
        self.decoder4 = DecoderBlock(64)

        self.conv = layers.Conv2D(filters=num_classes, kernel_size=1, activation=tf.nn.softmax)
    
    def call(self, inputs, training=None):


        x1, p1 = self.encoder1(inputs)
        x2, p2 = self.encoder2(p1)
        x3, p3 = self.encoder3(p2)
        x4, p4 = self.encoder4(p3)

        x = self.cb(p4)

        x = self.decoder1(x, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)

        x = self.conv(x)
        return x
    
    def summary(self):
        x = Input(shape=(self.width, self.height, 3))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()