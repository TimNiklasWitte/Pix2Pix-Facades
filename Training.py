import tensorflow as tf
import tqdm
import datetime

from Pix2Pix import *

NUM_EPOCHS = 300
BATCH_SIZE = 2

num_imgs_tensorboard = 16
interval_log_imgs_tensorboard = 10

def main():

    #
    # Load dataset
    #

    train_ds = tf.keras.utils.image_dataset_from_directory(
            validation_split=0.1,
            subset="training",
            directory="./facades/",
            labels=None,
            batch_size=None,
            seed=123)
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
            validation_split=0.1,
            subset="validation",
            directory="./facades/",
            labels=None,
            batch_size=None,
            seed=123)
    
    train_ds = train_ds.apply(prepare_data)
    test_ds = test_ds.apply(prepare_data)

    #
    # Initialize model.
    #

    pix2pix = Pix2Pix()

    # Build
    pix2pix.generator.build(input_shape=((1,256,256,3)))
    pix2pix.discriminator.build(input_shape=((1,256,256,6)))

    # Print number of parameters
    pix2pix.generator.summary()
    pix2pix.discriminator.summary()
    

    #
    # Logging
    #

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    #
    # 
    #

    drawing_imgs_list = []
    real_imgs_list = [] 
    for drawing_img, real_img in test_ds.take(num_imgs_tensorboard):
        drawing_imgs_list.append(drawing_img)
        real_imgs_list.append(real_img)

    drawing_imgs = tf.concat(drawing_imgs_list, axis=0)[:num_imgs_tensorboard]
    real_imgs = tf.concat(real_imgs_list, axis=0)[:num_imgs_tensorboard]

    log(train_summary_writer, pix2pix, drawing_imgs, real_imgs, epoch = 0)

    #
    # Train loop
    #
    for epoch in range(1, NUM_EPOCHS + 1):
            
        print(f"Epoch {epoch}")

        for drawing_img, real_img in tqdm.tqdm(train_ds, position=0, leave=True):
            pix2pix.train_step(drawing_img, real_img)

        log(train_summary_writer, pix2pix, drawing_imgs, real_imgs, epoch)

        if epoch % 25 == 0:
            # Save model (its parameters)
            pix2pix.generator.save_weights(f"./saved_models/generator/trained_weights_{epoch}", save_format="tf")
            pix2pix.discriminator.save_weights(f"./saved_models/discriminator/trained_weights_{epoch}", save_format="tf")


def log(train_summary_writer, pix2pix, drawing_imgs, real_imgs, epoch):

    #
    # Generate images
    #

    generated_imgs = pix2pix.generator(drawing_imgs, training=False)

    if epoch % interval_log_imgs_tensorboard == 0:
        imgs = tf.concat([drawing_imgs, generated_imgs, real_imgs], axis=2)
        imgs = (imgs + 1)/2

    num_generated_imgs = generated_imgs.shape[0]
    with train_summary_writer.as_default():

        for metric in pix2pix.generator.metrics:
            tf.summary.scalar(f"generator_{metric.name}", metric.result(), step=epoch)
            print(f"generator_{metric.name}: {metric.result()}")
            metric.reset_state()

        for metric in pix2pix.discriminator.metrics:
            tf.summary.scalar(f"discriminator_{metric.name}", metric.result(), step=epoch)
            print(f"discriminator_{metric.name}: {metric.result()}")
            metric.reset_state()

        if epoch % interval_log_imgs_tensorboard == 0:
            tf.summary.image(name="generated_imgs",data = imgs, step=epoch, max_outputs=num_generated_imgs)




def random_mirror(drawing, real):
    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        drawing = tf.image.flip_left_right(drawing)
        real = tf.image.flip_left_right(real)
    
    return drawing, real


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
    # random crop
    #

    dataset = dataset.map(lambda drawing, real: tf.stack([drawing, real], axis=0))
    dataset = dataset.map(lambda stacked_imgs: tf.image.random_crop(
      stacked_imgs, size=[2, 256, 256, 3]))
    

    dataset = dataset.map(lambda stacked_imgs: (stacked_imgs[0], stacked_imgs[1]))

    # Mirror
    dataset = dataset.map(lambda drawing, real: random_mirror(drawing, real))

    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")