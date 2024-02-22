import tensorflow as tf

from Generator import *
from Discriminator import *

class Pix2Pix(tf.keras.Model):

    def __init__(self):
        super(Pix2Pix, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.bce_loss = tf.keras.losses.BinaryCrossentropy()
        self.mae_loss = tf.keras.losses.MeanAbsoluteError()

        self.LAMBDA = 100

    @tf.function
    def train_step(self, drawing_img, real_img):

        with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2:
            fake_img = self.generator(drawing_img, training=True)

            drawing_concat_real = tf.concat([drawing_img, real_img], axis=-1)
            drawing_concat_fake = tf.concat([drawing_img, fake_img], axis=-1)


            pred_fake_img = self.discriminator(drawing_concat_fake, training=True)
            pred_real_img = self.discriminator(drawing_concat_real, training=True)

            generator_classic_loss = self.bce_loss(tf.ones_like(pred_fake_img), pred_fake_img)
            l1_loss = self.mae_loss(real_img, fake_img)

            generator_loss = generator_classic_loss + (self.LAMBDA * l1_loss)

            #
            # Discriminator
            #
            discriminator_fake_loss = self.bce_loss(tf.zeros_like(pred_fake_img), pred_fake_img)
            discriminator_real_loss = self.bce_loss(tf.ones_like(pred_real_img), pred_real_img)

            discriminator_loss = discriminator_fake_loss + discriminator_real_loss

        
        # Update generator
        gradients = tape_1.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        

        # Update discriminator
        gradients = tape_2.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        #
        # Update metrices
        #

        # Generator
        self.generator.metric_loss.update_state(generator_loss)
        self.generator.metric_classic_loss.update_state(generator_classic_loss)
        self.generator.metric_l1_loss.update_state(l1_loss)
        
        # Discriminator

        # Loss
        self.discriminator.metric_real_loss.update_state(discriminator_real_loss)
        self.discriminator.metric_fake_loss.update_state(discriminator_fake_loss)

        self.discriminator.metric_loss.update_state(discriminator_loss)

        # Accuracy
        classified_real = tf.math.round(pred_real_img)
        classified_fake = tf.math.round(pred_fake_img)
        
        ones = tf.ones_like(classified_real)
        zeros = tf.zeros_like(classified_fake)
        
        self.discriminator.metric_real_accuracy.update_state(ones, classified_real)
        self.discriminator.metric_fake_accuracy.update_state(zeros, classified_fake)

        

        