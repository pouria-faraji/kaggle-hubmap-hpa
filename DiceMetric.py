import tensorflow as tf

class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coefficient', smooth=1e-6, **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice_coefficient = self.add_weight(name='dice_coefficient', initializer='zeros')
        self.smooth = smooth

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred) + self.smooth
        denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth
        self.dice_coefficient.assign(2 * intersection / denominator)

    def result(self):
        return self.dice_coefficient

    def reset_state(self):
        self.dice_coefficient.assign(0.)