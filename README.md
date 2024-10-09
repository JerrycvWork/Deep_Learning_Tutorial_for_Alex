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

(02/10 Last Modification;)

### Prediction for Nerve Classification

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```
You can specify which model file to use with `--model MODEL.pth`.



### Training for Nerve Segmentation

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```


### Prediction for Nerve Segmentation

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```
You can specify which model file to use with `--model MODEL.pth`.


