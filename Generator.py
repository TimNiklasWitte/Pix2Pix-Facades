import tensorflow as tf

import tensorflow as tf

from DownSampleLayer import *
from UpSampleLayer import *

class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        self.down_sample_layer_stack = [
            DownSampleLayer(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            DownSampleLayer(128, 4),  # (batch_size, 64, 64, 128)
            DownSampleLayer(256, 4),  # (batch_size, 32, 32, 256)
            DownSampleLayer(512, 4),  # (batch_size, 16, 16, 512)
            DownSampleLayer(512, 4),  # (batch_size, 8, 8, 512)
            DownSampleLayer(512, 4),  # (batch_size, 4, 4, 512)
            DownSampleLayer(512, 4),  # (batch_size, 2, 2, 512)
            DownSampleLayer(512, 4),  # (batch_size, 1, 1, 512)
        ]

        self.up_sample_layer_stack = [
            UpSampleLayer(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            UpSampleLayer(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            UpSampleLayer(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            UpSampleLayer(512, 4),  # (batch_size, 16, 16, 1024)
            UpSampleLayer(256, 4),  # (batch_size, 32, 32, 512)
            UpSampleLayer(128, 4),  # (batch_size, 64, 64, 256)
            UpSampleLayer(64, 4),  # (batch_size, 128, 128, 128)
        ]

        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.last = tf.keras.layers.Conv2DTranspose(3, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=self.initializer,
                                                activation='tanh')  # (batch_size, 256, 256, 3)
  

    
        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_classic_loss = tf.keras.metrics.Mean(name="classic_loss")
        self.metric_l1_loss = tf.keras.metrics.Mean(name="l1_loss")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)


    @tf.function
    def call(self, x, training=False):

        #
        # Downsampling
        #
        skips = []
        for layer in self.down_sample_layer_stack:
            x = layer(x, training)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for layer, skip in zip(self.up_sample_layer_stack, skips):
            x = layer(x, training)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = self.last(x)

        return x

