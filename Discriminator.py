import tensorflow as tf

from DownSampleLayer import *
from UpSampleLayer import *

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_list = [
            DownSampleLayer(64, 4, False),
            DownSampleLayer(128, 4),
            DownSampleLayer(256, 4)
        ]

        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1,use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()  
        self.last = tf.keras.layers.Conv2D(1, 4, strides=1, activation="sigmoid")


        self.metric_fake_loss = tf.keras.metrics.Mean(name="fake_loss")
        self.metric_real_loss = tf.keras.metrics.Mean(name="real_loss")
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

        self.metric_real_accuracy = tf.keras.metrics.Accuracy(name="real_accuracy")
        self.metric_fake_accuracy = tf.keras.metrics.Accuracy(name="fake_accuracy")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)



    @tf.function
    def call(self, x, training=False):

        for layer in self.layer_list:
            x = layer(x, training)

        x = self.conv(x)
        x = self.batchnorm1(x, training)
        x = self.leaky_relu(x)
        x = self.zero_pad2(x)

        x = self.last(x)


        return x
