import os
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network.metrics import recall, precision, f1_score
from util.config import OutputConfig
from util.hdf5 import IMAGES_KEY, LABELS_KEY

BINARIZATION_THRESHOLD = 0.5
LABEL_COLOR = 0.5

def save_predictions(predictions: tf.Tensor, output_config: OutputConfig) -> None:
    """Save the predictions plainly as just images."""
    for idx, prediction in enumerate(predictions):
        prediction = tf.cast((prediction > BINARIZATION_THRESHOLD) * 255 * LABEL_COLOR, tf.uint8)
        img = tf.keras.preprocessing.image.array_to_img(prediction, scale=False)
        img.save(os.path.join(output_config.predictions_dir, f'{idx}.png'))

def visualize_prediction_comparisons(predictions: tf.Tensor, output_config: OutputConfig, dilate_labels: bool) -> None:
    """Visualize the predictions into a comparison between the image, the ground truth and the predicted label."""

    data_file = h5py.File(output_config.validation_set_file, 'r')
    num_images = len(data_file[IMAGES_KEY])

    # Preprocess the images and labels in bulk
    images = np.flip(np.array(data_file[IMAGES_KEY][:] * 255, dtype=np.uint8), axis=-1)
    labels = np.array(data_file[LABELS_KEY][:]).squeeze() * LABEL_COLOR

    label_tensors = tf.convert_to_tensor(np.array(data_file[LABELS_KEY][:]), dtype=tf.float32)
    prediction_labels = ((predictions.squeeze() > BINARIZATION_THRESHOLD) * 1) * LABEL_COLOR
    
    # Loop over the images and produce a plot with original image, ground truth and prediction
    for image_index in range(num_images):
        plot_file = os.path.join(output_config.predictions_dir, f'{image_index}.png')

        # Calculate and format metrics
        y_true = tf.expand_dims(label_tensors[image_index], 0)
        y_pred = tf.expand_dims(predictions[image_index], 0)

        recall_value = float(recall(y_true, y_pred, dilate_labels))
        precision_value = float(precision(y_true, y_pred, dilate_labels))
        f1_score_value = float(f1_score(y_true, y_pred, dilate_labels))

        recall_value = int(round(recall_value, 2) * 100)
        precision_value = int(round(precision_value, 2) * 100)
        f1_score_value = int(round(f1_score_value, 2) * 100)

        # Create figure with 3 subplots
        fig = plt.figure()
        fig.set_size_inches((3, 3))
        ax1 = plt.subplot2grid((1,1), (0,0))
        divider = make_axes_locatable(ax1) 
        ax2 = divider.append_axes("bottom", size="100%", pad=0.1)
        ax3 = divider.append_axes("bottom", size="100%", pad=0.4)
        
        # Show the images
        ax1.imshow(images[image_index])
        ax2.imshow(labels[image_index], vmin=0, vmax=1, cmap='gray')
        ax3.imshow(prediction_labels[image_index], vmin=0, vmax=1, cmap='gray')
        
        # Show the scores
        ax3.set_title(f'F1:{f1_score_value}% / RE:{recall_value}%\nPR:{precision_value}%', fontsize=7)
        
        # Remove axes
        ax1.axis('off')  
        ax2.axis('off')  
        ax3.axis('off') 

        # Save
        plt.tight_layout()
        plt.savefig(plot_file, bbox_inches = 'tight', dpi=100, pad_inches=0.05)
        plt.close()
        