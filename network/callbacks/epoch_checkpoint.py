from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
	"""Callback which saves checkpoints every couple of epochs."""

	checkpoints_dir: str
	save_weights_only: bool
	interval: int
	start_epoch: int

	def __init__(self, checkpoints_dir: str, save_weights_only: bool, interval: int, start_epoch: int):
		super(Callback, self).__init__()

		self.checkpoints_dir = checkpoints_dir
		self.save_weights_only = save_weights_only
		self.interval = interval
		self.start_epoch = start_epoch

	def on_epoch_end(self, epoch: int, logs: dict = {}) -> None:
		"""Serialize the model every interval epochs."""
		if epoch % self.interval == 0:
			if self.save_weights_only:
				self.model.save_weights(os.path.join(self.checkpoints_dir, f'epoch_{epoch + 1}.weights.h5'))
			else:
				self.model.export(os.path.join(self.checkpoints_dir, f'epoch_{epoch + 1}_model.keras'))
