import tensorflow as tf
# from tensorflow.python.keras import Model, layers, Sequential, Input, regularizers
from tensorflow.keras import Model, layers, Sequential, Input, regularizers

import segmentation_models as sm
# Segmentation Models: using `keras` framework.

class ResNet(Model):
    BACKBONE = 'resnet34'
    def __init__(self, num_classes=2, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height
        self.num_classes = num_classes

        self.unet_layer = sm.Unet(self.BACKBONE,
                                  classes=self.num_classes,
                                  encoder_weights='imagenet',
                                  activation='softmax',
                                  encoder_freeze=True)

    def call(self, inputs, training=None):
        return self.unet_layer(inputs)

    def summary(self):
        x = Input(shape=(self.width, self.height, 3))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()