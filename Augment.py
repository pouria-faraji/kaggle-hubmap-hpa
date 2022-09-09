import tensorflow as tf


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.flip_inputs_1 = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.flip_labels_1 = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

        self.flip_inputs_2 = tf.keras.layers.RandomFlip(mode="vertical", seed=seed)
        self.flip_labels_2 = tf.keras.layers.RandomFlip(mode="vertical", seed=seed)

        self.rotate_inputs = tf.keras.layers.RandomRotation(factor=0.2, seed=seed)
        self.rotate_labels = tf.keras.layers.RandomRotation(factor=0.2, seed=seed)

        # self.contrast_input = tf.keras.layers.RandomContrast(factor=0.4, seed=seed)


    def call(self, inputs, labels):
        inputs = self.flip_inputs_1(inputs)
        labels = self.flip_labels_1(labels)

        inputs = self.flip_inputs_2(inputs)
        labels = self.flip_labels_2(labels)

        inputs = self.rotate_inputs(inputs)
        labels = self.rotate_labels(labels)

        # inputs = self.contrast_input(inputs)

        return inputs, labels
