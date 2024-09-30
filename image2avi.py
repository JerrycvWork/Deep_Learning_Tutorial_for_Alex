#Video Generation With Ground Truth

import os
import glob
import imageio
from natsort import natsorted, ns
import cv2
import numpy as np



png_dir = './output2/'

images = []
file_names = glob.glob(png_dir+"Opening Wound__*.jpg")
print('before sort',file_names)
file_names = natsorted(file_names, key=lambda y: y.split("_")[-1].lower())
print('sorted',file_names)

name = os.path.dirname(os.path.realpath(__file__))

for file_name in file_names:
    img = cv2.imread(file_name)
    height, width, layers = img.shape
    size = (width,height)
    images.append(img)
    print(file_name)

file_names = glob.glob(png_dir+"Tendon*.jpg")
print('before sort',file_names)
file_names = natsorted(file_names, key=lambda y: y.split("_")[-1].lower())
print('sorted',file_names)

name = os.path.dirname(os.path.realpath(__file__))

for file_name in file_names:
    img = cv2.imread(file_name)
    height, width, layers = img.shape
    size = (width,height)
    images.append(img)
    print(file_name)

file_names = glob.glob(png_dir+"Opening Wound to nerve*.jpg")
print('before sort',file_names)
file_names = natsorted(file_names, key=lambda y: y.split("_")[-1].lower())
print('sorted',file_names)

name = os.path.dirname(os.path.realpath(__file__))

for file_name in file_names:
    img = cv2.imread(file_name)
    height, width, layers = img.shape
    size = (width,height)
    images.append(img)
    print(file_name)

file_names = glob.glob(png_dir+"Nerve*.jpg")
print('before sort',file_names)
file_names = natsorted(file_names, key=lambda y: y.split("_")[-1].lower())
print('sorted',file_names)

name = os.path.dirname(os.path.realpath(__file__))

for file_name in file_names:
    img = cv2.imread(file_name)
    height, width, layers = img.shape
    size = (width,height)
    images.append(img)
    print(file_name)

file_names = glob.glob(png_dir+"Transection of nerve"+"*.jpg")
print('before sort',file_names)
file_names = natsorted(file_names, key=lambda y: y.split("_")[-1].lower())
print('sorted',file_names)

name = os.path.dirname(os.path.realpath(__file__))

for file_name in file_names:
    img = cv2.imread(file_name)
    height, width, layers = img.shape
    size = (width,height)
    images.append(img)
    print(file_name)

out = cv2.VideoWriter('sample_v2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1.6, size)

for i in range(len(images)):
    out.write(images[i])
out.release()


# Opening wound, Tendon, Opening Wound to Nerve, Nerve, Transaction to Nerve