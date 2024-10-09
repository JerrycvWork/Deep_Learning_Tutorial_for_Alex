# Deep Learning Tutorial (Classification and Segmentation for nerve image) for Alex

Customized implementation and Deep Learning Tutorial for Nerve Classification and Segmentation. For Alex.

- [Quick start](#Quick-start)
- [Deep Learning Step for Classification and Segmentation](#Deep-Learning-Step-for-Classification-and-Segmentation)
- [Usage](#usage)
  - [Dataset](#Dataset)
  - [Training for Nerve Classification](#Training-for-Nerve-Classification)
  - [Prediction for Nerve Classification](#Prediction-for-Nerve-Classification)
  - [Training for Nerve Segmentation](#Training-for-Nerve-Segmentation)
  - [Prediction for Nerve Segmentation](#Prediction-for-Nerve-Segmentation)

## Quick start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch in GPU version](https://pytorch.org/get-started/locally/)

3. Install dependencies

```bash
pip install -r requirements.txt
```


## Deep Learning Step for Classification and Segmentation

Input: Image

(The Step of writing code for classification and segmentation)

The general strucure of writing the code for training deep learning network could refer the `train_cls.py` and `train_seg.py` in its main functions (From Line 61 to end).

### Classification Step

Classification (training, Please refer the `train_cls.py` as the sample)

Step1: Pre-define the parameters of training/dataset/path for saving the checkpoints.

Step2: Define the dataset in both training and testing. (Exp: `Line 59-72`) 

The definition of dataset often contain the input image size, the batch size and if shuffle the datasets randomly.

Moreover, some framework would define the data augmentation after the definition of datasedet. Exp: `Line 91-94`. The random flip and random rotation are commonly used for data augmentation.

Step3: Define the deep learning model for training. Some works would define the model structure in another python files and call them in the main function. In this sample, we apply the 

base model as the encoder and add other layers in the main function. Please see `Line 99-118` for the details of model construction.

Step4: Set the Optimizer and corresponding loss function for training. Different deep learning platforms have different methods to set. In Tensorflow/Keras, please see `Line 133-135` for compiling the model.

Step5: 


Segmentation

Step1:

Step2:


## Usage

### Dataset

Wait for the filling.

### Training for Nerve Classification

The script for training the classification network for Nerve is `train_cls.py`.

The parameters of training setting is from line 19 to line 33. Please set them before starting the training.


```console

> python train_cls.py

```


### Training for Nerve Segmentation

The script for training the segmentation network for Nerve is `train_seg.py`. (Except the network MultiResUNet, this network is in `train_multiresunet.py` which share the similar code structure)

The parameters of training setting is from line 85 to line 89. The path for datasets is from line 70-71. Please set them before starting the training.


```console

> python train_seg.py

```


### Prediction for Nerve Segmentation

After training your model in both classification and segmentation model and saving it to `folder+'/model_path'` and `"files/model.h5"`, you can easily test the output masks on your images with nerve classification and segemntation.

The running of prediction is written in `link_copy.py`. We provide the example data for testing the code for running in folder `example_data`. Please unzip in the root of code.

After that, please fill the path for getting the output in line 35. And also modify the model path of classification and segmentation in line 274 and 275.

After that, the result could be found in the output folder.

(08/10 Last Modification)


