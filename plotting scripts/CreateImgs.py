import sys
sys.path.append("../")

import tensorflow as tf
import tqdm

from Pix2Pix import *
from matplotlib import pyplot as plt



def main():

    test_ds = tf.keras.utils.image_dataset_from_directory(
            validation_split=0.1,
            subset="validation",
            directory="../facades/",
            labels=None,
            batch_size=None,
            seed=123)

    test_ds = test_ds.apply(prepare_data) 

    pix2pix = Pix2Pix() 

    pix2pix.generator.build(input_shape=(1, 256, 256, 3))
    pix2pix.discriminator.build(input_shape=((1,256,256,6)))

    pix2pix.generator.load_weights(f"../saved_models/generator/trained_weights_300").expect_partial()
    pix2pix.discriminator.load_weights(f"../saved_models/discriminator/trained_weights_300").expect_partial()

    
    for idx, (drawing_img, real_img) in enumerate(tqdm.tqdm(test_ds, position=0, leave=True)):

        fig = plt.figure()
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, 0])

        ax2_1 = fig.add_subplot(gs[0, 1])

        ax2_2 = fig.add_subplot(gs[1, 1])

        ax3_1 = fig.add_subplot(gs[0, 2])

        ax3_2 = fig.add_subplot(gs[1, 2])

        generated_img = pix2pix.generator(drawing_img, training=False)


        drawing_concat_real = tf.concat([drawing_img, real_img], axis=-1)
        drawing_concat_fake = tf.concat([drawing_img, generated_img], axis=-1)


        drawing_img = drawing_img[0]
        generated_img = generated_img[0]
        real_img = real_img[0]

        drawing_img = (drawing_img + 1)/2
        generated_img = (generated_img + 1)/2
        real_img = (real_img + 1)/2


        pred_fake_img = pix2pix.discriminator(drawing_concat_fake, training=False)
        pred_fake_img = pred_fake_img[0]

        pred_real_img = pix2pix.discriminator(drawing_concat_real, training=False)
        pred_real_img = pred_real_img[0]

        ax1.imshow(drawing_img)
        ax1.axis("off")
        ax1.set_title("Input")

        
        ax2_1.imshow(generated_img)
        ax2_1.axis("off")
        ax2_1.set_title("Generated image")

        ax2_2.imshow(pred_fake_img)
        ax2_2.axis("off")
        pred_fake_img = tf.math.reduce_mean(pred_fake_img)
        ax2_2.set_title(f"Discriminator patches\n(avg: {pred_fake_img:1.3})")

        

        ax3_1.imshow(real_img)
        ax3_1.axis("off")
        ax3_1.set_title("Ground truth")

        
        ax3_2.imshow(pred_real_img)
        ax3_2.axis("off")
        pred_real_img = tf.math.reduce_mean(pred_real_img)
        ax3_2.set_title(f"Discriminator patches\n(avg: {pred_real_img:1.3f})")
      

        plt.tight_layout()
        plt.savefig(f"../plots/generated images/{idx}.png", bbox_inches='tight')
        plt.close()

       


def prepare_data(dataset):

    dataset = dataset.map(lambda img: tf.image.resize(img, [256,512]))

    # Convert data from uint8 to float32
    dataset = dataset.map(lambda img: tf.cast(img, tf.float32) )

    # Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img: (img/128.)-1. )

    # Image contains input and target -> Split them
    width = 256
    dataset = dataset.map(lambda img: (img[:, width:, :], img[:, :width, :]))

    # Cache
    dataset = dataset.cache()

    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")