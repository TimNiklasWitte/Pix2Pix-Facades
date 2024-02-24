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

    pix2pix.generator.load_weights(f"../saved_models/generator/trained_weights_300").expect_partial()
   
    for idx, (drawing_img, real_img) in enumerate(tqdm.tqdm(test_ds, position=0, leave=True)):

        fig, axes = plt.subplots(nrows=1, ncols=2)

        generated_img = pix2pix.generator(drawing_img, training=False)

        drawing_img = drawing_img[0]
        generated_img = generated_img[0]
        real_img = real_img[0]

        drawing_img = (drawing_img + 1)/2
        generated_img = (generated_img + 1)/2
        real_img = (real_img + 1)/2

        axes[0].imshow(drawing_img)
        axes[0].axis("off")
        axes[0].set_title("Input")

        axes[1].imshow(generated_img)
        axes[1].axis("off")
        axes[1].set_title("Generated image")

        plt.tight_layout()
        plt.savefig(f"../plots/results/{idx}.png", bbox_inches='tight')
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