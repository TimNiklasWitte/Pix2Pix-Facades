import tensorflow as tf

class UpSampleLayer(tf.keras.layers.Layer):

    def __init__(self, filters, size, apply_dropout=False):
        super(UpSampleLayer, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.layer_list = [
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False),
            tf.keras.layers.BatchNormalization(),

        ]

        if apply_dropout:
            self.layer_list.append(tf.keras.layers.Dropout(0.5))

        self.layer_list.append(tf.keras.layers.ReLU())

    def call(self, x, training):
        for layer in self.layer_list:
            
            if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training)
            else:
                x = layer(x)
        
        return x