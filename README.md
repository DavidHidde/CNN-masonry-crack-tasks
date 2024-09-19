# Crack segmentation for masonry surfaces

This is a fork of the [crack_detection_CNN_masonry repoistory](https://github.com/dimitrisdais/crack_detection_CNN_masonry), which aims to share the code of their research about crack detection/segmentation on masonry surfaces. This fork improves upon the original code through the following:

* Provide a clearer, well-documented codebase which is easier to interpret and use.
* Separation of the input parameters into configuration files, making it significantly easier to run, train and test multiple networks at the same time.
* Remove weird and unnecessary conventions like folder naming.
* Add better generalizability through the use of configuration files and enhanced dataset support. Add support for more segmentation models.
* Dependency cleanup: clearly indicate all dependencies through a `requirements.txt` file and make the framework compatible with modern Tensorflow.
* Add QOL features such as test set creation and resuming training.
* Fix errors in the original code.

For changes compared to the original repo, please have a look at the [PR descriptions](https://github.com/DavidHidde/CNN-masonry-crack-tasks/pulls?q=is%3Apr+is%3Aclosed+). The main changes include:

* Standardized loss computations. Dice, focal and WCE loss now utilize the direct formulas or Keras/TF functionality.
* Fixes F1-score computation. The original repo can have F1-scores above 1 if dilated due to an error in the computation.
* Unet now only uses the segmentation models variant rather than a custom one depending on the use of a backbone. The Unet parameters have also been updated to use transpose layers in the decoder, matching the paper.
* DeepCrack uses a new [implementation](https://github.com/DavidHidde/DeepCrack-Tensorflow).
* Newer package versions, making the code compatible with modern systems and Python versions.

Overall, the core functionality is retained but updated to work with modern setups, be more flexible as well as maintainable. Some results might differ from the original paper, but it is clear that the forked repo does not fully represent the original paper due to its broken state and lack of crack classification support.

![cover_image](images/crack_detection.png)

# Installation

The project makes use of multiple dependencies. To install these, simply run `pip3 install -r requirements.txt`.  

Like the original repo, some files must currently be copied over from other repos in order for some configurations to work. This repo solves it through Git submodules:

```bash
git submodule init --recursive
```

# Usage

The basic entrypoint of the program is `run_model.py`:

```bash
python3 run_model.py --operation OPERATION
```

where `OPERATION` should indicate the type of operation to perform. The operation types consist of:

* `build`: Build and process the dataset into a training and validation set.
* `train`: Train a network.
* `test`: Generate predictions.
* `visualize`: Visualize the model architecture in a file.

All operations are listed in their respective [directory](operations/implementation).

## Configuration files

Examples: [`example_network_config.yaml`](example_network_config.yaml) and [`example_dataset_config.yaml`](example_dataset_config.yaml)  

Note that many of the configurations options are either strings, numbers or booleans. These speak for themselves. For some settings, only a set of values are possible. These values are listed in [`types.py`](util/types.py). An overview of all settings and their types is provided in the [`configs`](util/config/).

## Basic use case example

A basic example for how this repository functions is described here. 

### Building the dataset

Given the task to train a segmentation network on a dataset, you should first start by building the dataset.
This is done by first copying the dataset files to a new folder inside the [dataset](dataset) directory and then copying [`example_dataset_config.yaml`](example_dataset_config.yaml) and inserting the correct dataset values.
Once you have split the images and labels into separate folders and created the dataset config, you can simply call:

```bash
python3 run_model.py -o build -d your_dataset_config.yaml 
```

This should build the `train.hdf5` and `val.hdf5` inside the dataset directory for you.

### Training a model

The next step is to train a model. The first step is to copy [`example_network_config.yaml`](example_network_config.yaml) and determine the network you want to train.
After you have decided this, only 1 simple call is needed:

```bash
python3 run_model.py -o train -d your_dataset_config.yaml -n your_network_config.yaml
```

This should train the model for you. In case something happens and the training interrupts, you can choose to resume the model using:

```bash
python3 run_model.py -o train -d your_dataset_config.yaml -n your_network_config.yaml --weights name_of_the_newest_checkpoint.keras
```

### Testing the trained model

Finally, to test the model one can simply run:

```bash
python3 run_model.py -o test -d your_dataset_config.yaml -n your_network_config.yaml
```

This should provide you with some prediction files which visualize the difference between the ground truth and predicted label.

# Acknowledgements - original work

The original repository was produced to share material relevant to the Journal paper **[Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning](https://doi.org/10.1016/j.autcon.2021.103606)** by **[D. Dais](https://www.researchgate.net/profile/Dimitris-Dais)**,  **İ. E. Bal**, **E. Smyrou**, and **V. Sarhosis** published in **Automation in Construction**.  

The paper can be downloaded from the following links:
- [https://doi.org/10.1016/j.autcon.2021.103606](https://doi.org/10.1016/j.autcon.2021.103606)
- [https://www.researchgate.net/publication/349645935_Automatic_crack_classification_and_segmentation_on_masonry_surfaces_using_convolutional_neural_networks_and_transfer_learning
](https://www.researchgate.net/publication/349645935_Automatic_crack_classification_and_segmentation_on_masonry_surfaces_using_convolutional_neural_networks_and_transfer_learning)

In case you use or find interesting their work please cite the following journal publication:

**D. Dais, İ.E. Bal, E. Smyrou, V. Sarhosis, Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning, Automation in Construction. 125 (2021), pp. 103606. https://doi.org/10.1016/j.autcon.2021.103606.**

``` 
@article{Dais2021,  
  author = {Dais, Dimitris and Bal, İhsan Engin and Smyrou, Eleni and Sarhosis, Vasilis},  
  doi = {10.1016/j.autcon.2021.103606},  
  journal = {Automation in Construction},  
  pages = {103606},  
  title = {{Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning}},  
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0926580521000571},  
  volume = {125},  
  year = {2021}  
}  
```

# References

The following codes are based on material provided by **[Adrian Rosebrock](linkedin.com/in/adrian-rosebrock-59b8732a)** shared on his blog (**https://www.pyimagesearch.com/**) and his books:

* `hdf5_dataset_generator.py`  
* `hdf5_dataset_writer.py`
* `epoch_checkpoint.py`
* `training_monitor.py`

- Adrian Rosebrock, Deep Learning for Computer Vision with Python - Practitioner Bundle, PyImageSearch, https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/, accessed on 24 February 2021  
- Adrian Rosebrock, Keras: Starting, stopping, and resuming training, PyImageSearch, https://www.pyimagesearch.com/2019/09/23/keras-starting-stopping-and-resuming-training/, accessed on 24 February 2021  
- Adrian Rosebrock, How to use Keras fit and fit_generator (a hands-on tutorial), PyImageSearch, https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/, accessed on 24 February 2021  

The Segmentation Models with pre-trained CNNs are implemented based on the work of **[Pavel Yakubovskiy](https://github.com/qubvel)** and his GitHub Repository https://github.com/qubvel/segmentation_models
