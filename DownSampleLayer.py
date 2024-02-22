import tensorflow as tf

class DownSampleLayer(tf.keras.layers.Layer):

    def __init__(self, filters, size, apply_batchnorm=True):
        super(DownSampleLayer, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.layer_list = [
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False)
        ]

        if apply_batchnorm:
            self.layer_list.append(tf.keras.layers.BatchNormalization())

        self.layer_list.append(tf.keras.layers.LeakyReLU())

    def call(self, x, training):
        for layer in self.layer_list:
            
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x,training)
            else:
                x = layer(x)
        
        return x