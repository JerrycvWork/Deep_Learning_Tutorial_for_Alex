#!/usr/bin/env python
# coding: utf-8

# Import the Library (Including Tensorflow, Numpy)
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from glob import glob
from sklearn.model_selection import train_test_split
from model import build_model
from utils import *
from metrics import *
print("GPU Usage Status: ",tf.test.is_gpu_available())


# Define the functions for reading image, etc
def read_image(x):
    x = x.decode()
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image=cv2.resize(image,(224,224))
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    return image

def read_mask(y):
    y = y.decode()
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (224, 224))
    mask = mask/255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def parse_data(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        y = np.concatenate([y, y], axis=-1)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([224, 224, 3])
    y.set_shape([224, 224, 2])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.map(map_func=parse_data)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch)
    return dataset

# Main Program

if __name__ == "__main__":

    # Could ignore. Just for the stable setting
    np.random.seed(42)
    tf.random.set_seed(42)

    # Dataset Setting
    create_dir("files")

    train_path = "/home/htihe/datadisk/Data_OLD/Nervesegmentation/Rearrange/train/"
    valid_path = "/home/htihe/datadisk/Data_OLD/Nervesegmentation/Rearrange/val/"

    ## Training Dataset
    train_x = sorted(glob(os.path.join(train_path, "image", "*.png")))
    train_y = sorted(glob(os.path.join(train_path, "mask", "*.png")))

    ## Shuffling
    train_x, train_y = shuffling(train_x, train_y)

    ## Validation Dataset
    valid_x = sorted(glob(os.path.join(valid_path, "image", "*.png")))
    valid_y = sorted(glob(os.path.join(valid_path, "mask", "*.png")))

    ## Model Setting
    model_path = "files/model.h5"
    batch_size = 3
    epochs = 20
    lr = 1e-4
    shape = (224, 224, 3)

    model = build_model(shape)
    metrics = [
        dice_coef,
        iou,
        Recall(),
        Precision()
    ]
    
    ## Dataset Setting
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    
    ## Model Complie and start the training
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False)
    ]

    train_steps = (len(train_x)//batch_size)
    valid_steps = (len(valid_x)//batch_size)

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model.fit(train_dataset,
            epochs=epochs,
            validation_data=valid_dataset,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=callbacks,
            shuffle=False)
    model.save("new.h5")




