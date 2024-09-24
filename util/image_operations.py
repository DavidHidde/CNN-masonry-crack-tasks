import keras
import tensorflow as tf


def dilation2d(img: tf.Tensor, kernel_size: int):
    """
    Function to dilate an image, used to dilate labels to create some tolerance.
    """
    with tf.name_scope('dilation2d'):
        kernel = tf.zeros((kernel_size, kernel_size, 1))
        return tf.nn.dilation2d(
            img,
            kernel,
            (1, 1, 1, 1),
            'SAME',
            'NHWC',
            (1, 1, 1, 1)
        )

def get_data_augmentor() -> keras.Sequential:
    """Get the data augmentation layers."""
    return keras.Sequential([
        keras.layers.RandomRotation(factor=20/360, fill_mode='nearest'),
        keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode='nearest'),
        keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode='nearest'),
        keras.layers.RandomFlip(mode='horizontal')
    ])