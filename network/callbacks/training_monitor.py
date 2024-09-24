from typing import Union, TextIO

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import json

from util.types import MetricType


class TrainingMonitor(Callback):
    """Logger which saves the metrics of the model every epoch."""

    figure_file_path: str
    metrics_file_path: str
    start_epoch: int
    metric: MetricType

    history: dict
    metrics_file: Union[TextIO, None]

    def __init__(self, figure_file: str, metrics_json_file: str, start_epoch: int, metric: MetricType):
        super(TrainingMonitor, self).__init__()
        self.figure_file_path = figure_file
        self.metrics_file_path = metrics_json_file
        self.start_epoch = start_epoch
        self.metric = metric

        self.history = {}
        self.metrics_file = None

    def on_train_begin(self, logs: dict = {}) -> None:
        """Load the previous history if we start at a epoch > 0"""
        if self.start_epoch > 0:
            with open(self.metrics_file_path, 'r') as metrics_file:
                self.history = json.load(metrics_file)
                # Slice off all history past the current point
                for key in self.history.keys():
                    self.history[key] = self.history[key][:self.start_epoch]

        self.metrics_file = open(self.metrics_file_path, 'w')

    def on_train_end(self, logs: dict = {}) -> None:
        """Close the file once we end."""
        self.metrics_file.close()

    def on_epoch_end(self, epoch: int, logs: dict = {}) -> None:
        """Append the current metric values, serialize them and plot them."""
        for metric, value in logs.items():
            current_metrics = self.history.get(metric, [])
            current_metrics.append(float(value))
            self.history[metric] = current_metrics
        self.metrics_file.seek(0)
        json.dump(self.history, self.metrics_file)

        # Start plotting from the second epoch
        if epoch > 0:
            x = range(epoch + 1)
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(x, self.history['loss'], 'k--', label='Loss - Train')
            plt.plot(x, self.history['val_loss'], 'r--', label='Loss - Validation')
            
            plt.plot(x, self.history[self.metric.value], 'k-', label=f'{self.metric.value} - Train')
            plt.plot(x, self.history[f'val_{self.metric.value}'], 'r-', label=f'{self.metric.value} - Validation')
            plt.title(f'Loss/{self.metric.value} progression (epoch {epoch})')

            ticks, _ = plt.xticks()
            num_ticks = min(len(ticks), epoch + 1)
            plt.xticks(np.linspace(0, epoch, num_ticks, dtype=int))

            plt.xlabel('Epoch #')
            plt.ylabel(f'Loss/{self.metric.value}')
            plt.legend()

            # Save the figure
            plt.savefig(self.figure_file_path)
            plt.close()
            