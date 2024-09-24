from typing import Generator, Union

import tensorflow as tf
import keras
import numpy as np
import h5py

from util.hdf5 import IMAGES_KEY, LABELS_KEY


class HDF5DatasetGenerator:
    """Generator class for providing the dataset to a model."""

    SHUFFLE_SEED = 2024
    BINARIZATION_THRESHOLD = 0.5

    data_file: h5py.File
    num_images: int
    batch_size: int

    shuffle: bool
    binarize_labels: bool

    data_augmentor: Union[keras.Sequential, None]

    def __init__(
        self,
        data_file_path: str,
        batch_size: int,
        shuffle: bool,
        binarize_labels: bool,
        data_augmentor
    ):
        self.batch_size = batch_size
        self.binarize_labels = binarize_labels
        self.shuffle = shuffle

        self.data_file = h5py.File(data_file_path, 'r+')
        self.num_images = len(self.data_file[IMAGES_KEY])

        self.data_augmentor = data_augmentor
        
    def __call__(self, passes=np.inf) -> Generator:
        """Generate data for a certain number of passes. By default, this is infinite passes."""
        pass_idx = 0
        while pass_idx < passes:
            for batch_idx in np.arange(0, self.num_images, self.batch_size):
                images = tf.convert_to_tensor(self.data_file[IMAGES_KEY][batch_idx: batch_idx + self.batch_size], dtype=tf.float32)
                labels = tf.convert_to_tensor(self.data_file[LABELS_KEY][batch_idx: batch_idx + self.batch_size], dtype=tf.float32)

                if self.data_augmentor is not None:
                    images = self.data_augmentor(images)
                    labels = self.data_augmentor(labels)

                if self.binarize_labels:
                    labels = tf.cast(labels > self.BINARIZATION_THRESHOLD, dtype=labels.dtype)
        
                yield images, labels

            pass_idx += 1

    def close(self):
        """Close the dataset file."""
        self.data_file.close()
		
