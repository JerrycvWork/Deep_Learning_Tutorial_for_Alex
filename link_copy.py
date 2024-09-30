import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import tensorflow as tf
import time
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *
from train import tf_dataset
import math
from natsort import natsorted, ns
from tensorflow.keras.preprocessing import image_dataset_from_directory



from metrics import iou
from utils import *







#validation_dir = os.path.join('Nerve_600ms/')
folder_name = "./example_data/"
validation_dir = os.path.join(folder_name+'/input/')
hd_dir = os.path.join(folder_name+'/input/')
mask_dir = os.path.join(folder_name+'/annotation/')
order_dir = os.path.join(folder_name+'/order/')
real_order = [os.path.splitext(os.path.basename(x))[0] for x in glob(order_dir+"*.jpg")]
#real_order.sort(key)
real_order=natsorted(real_order, key=lambda y: y.split("_")[-1].lower())
print("Real Order: ",real_order)

output_dir = os.path.join(folder_name+'/output2/')

BATCH_SIZE = 1
IMG_SIZE = (224,224)
org_size = (1080,1080)
maskRatio = 1080/1608
#org_maskCoordinate1 = (math.floor(270),math.floor(0))
org_maskCoordinate1 = (math.floor(0),math.floor(0))

org_maskCoordinate2 = (math.floor(1080),math.floor(1080))



org_maskSize = ((org_maskCoordinate2[0]-org_maskCoordinate1[0]),(org_maskCoordinate2[1]-org_maskCoordinate1[1]))

pre_sort=[os.path.splitext(os.path.basename(x))[0] for x in glob(validation_dir+"*/*.jpg")]
pre_sort=natsorted(pre_sort, key=lambda y: y.split("/")[-1].split("_")[-1].lower())
print("Pre_sort",pre_sort)

validation_name = tf.data.Dataset.from_tensor_slices(pre_sort)
validation_name = validation_name.batch(BATCH_SIZE)
print("Validation Name",validation_name)
validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
class_names = validation_dataset.class_names
print("Class Name: ",class_names)
#validation_dataset = tf.data.Dataset.zip((validation_dataset, validation_name))
#
mask_dataset = image_dataset_from_directory(mask_dir,
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
validation_hd_dataset = image_dataset_from_directory(hd_dir,
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=org_size)

print("class_names: ",validation_dataset.file_paths)

def read_image(x):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def read_image_array(x):
    image = cv2.cvtColor(np.float32(x), cv2.COLOR_BGR2RGB)
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def read_mask(y):
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def read_mask_array(y):
    # mask =  cv2.cvtColor(np.float32(y), cv2.COLOR_BGR2RGB)
    mask = y.astype(np.float32)
    # mask = mask/255.0
    mask = np.expand_dims(mask, axis=0)
    return mask


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def parse(y_pred):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred[..., -1]
    y_pred = y_pred.astype(np.float32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    return y_pred


def multiplyGreen(input):
    b, g, r = cv2.split(input)
    np.multiply(b, 0, out=b, casting="unsafe")
    np.multiply(g, 2, out=g, casting="unsafe")
    np.multiply(r, 0, out=r, casting="unsafe")
    after = cv2.merge([b, g, r])
    return after


m_seg = tf.keras.metrics.MeanIoU(num_classes=2)
iou_ary_seg = []


def evaluate_normal(model, x_data, y_data):
    THRESHOLD = 0.5
    total = []
    for i, (x, y) in tqdm(enumerate(zip(x_data, y_data)), total=len(x_data)):

        x = read_image(x)
        y = read_mask(y)
        _, h, w, _ = x.shape

        y_pred1 = parse(model.predict(x)[0][..., -2])
        y_pred2 = parse(model.predict(x)[0][..., -1])
        y_pred3 = parse(model.predict(x)[0][..., -1])
        y_pred3[y_pred3 < THRESHOLD] = 0
        y_pred3[y_pred3 > THRESHOLD] = 1
        m_seg.reset_states()
        m_seg.update_state(y, y_pred3)
        # print("y",y)
        # print("y_pred3",y_pred3)
        # print("num ",i," iou: ",m.result().numpy())
        iou_ary_seg.append(m_seg.result().numpy())
        line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x[0] * 255.0, line,
            mask_to_3d(y) * 255.0, line,
            mask_to_3d(y_pred1) * 255.0, line,
            mask_to_3d(y_pred2) * 255.0
        ]
        mask = np.concatenate(all_images, axis=1)

        cv2.imwrite(f"results/{i}.png", mask)


def evaluate_1(_model, x_data, x_hd_data, x_mask_data):
    THRESHOLD = 0.5
    x_data=cv2.resize(x_data,(256,256))
    x_mask_data = cv2.resize(x_mask_data, (256,256))
    x = read_image_array(x_data)
    x_hd = read_image_array(x_hd_data)
    x_mask = read_mask_array(x_mask_data)
    _, h, w, _ = x.shape
    # y_pred1 = parse(_model.predict(x)[0][..., -2])
    # y_pred2 = parse(_model.predict(x)[0][..., -1])
    start_time=time.time()
    y_pred3 = parse(_model.predict(x)[0][..., -1])
    time2=time.time()-start_time
    y_pred3[y_pred3 < THRESHOLD] = 0
    y_pred3[y_pred3 > THRESHOLD] = 1
    _iouValue = iou(x_mask[0], mask_to_3d(y_pred3))
    print('x_mask[0]:', x_mask[0].shape)
    print('mask_to_3d(y_pred3)', mask_to_3d(y_pred3).shape)
    # src1= cv2.cvtColor(np.asarray(x[0] * 255.0),cv2.COLOR_RGB2BGR)
    src1 = cv2.cvtColor(np.asarray(x_hd_data), cv2.COLOR_RGB2BGR)
    print('src1.shape', src1.shape)
    src2 = cv2.cvtColor(np.asarray(mask_to_3d(y_pred3) * 255.0), cv2.COLOR_RGB2BGR)
    src2 = cv2.resize(src2, org_maskSize, interpolation=cv2.INTER_CUBIC)
    src2 = multiplyGreen(src2)
    x_offset = org_maskCoordinate1[0]
    y_offset = org_maskCoordinate1[1]
    dst = src1
    print('dst.shape', dst.shape)
    print('src2.shape', src2.shape)
    print(src2[0, 0])
    cv2.imwrite("x_mask[0].png", cv2.cvtColor(np.asarray(x_mask[0]) * 255, cv2.COLOR_RGB2BGR))
    cv2.imwrite("mask_to_3d(y_pred3).png", cv2.cvtColor(np.asarray(mask_to_3d(y_pred3)) * 255, cv2.COLOR_RGB2BGR))
    dst[y_offset:y_offset + src2.shape[0], x_offset:x_offset + src2.shape[1]] = src2 * 0.2 + dst[y_offset:y_offset +
                                                                                                          src2.shape[0],
                                                                                             x_offset:x_offset +
                                                                                                      src2.shape[
                                                                                                          1]] * 0.8
    # dst = cv2.addWeighted(src1, 0.8, src2, 0.2, 0)
    return dst, _iouValue,time2


def predict_1(_model, x_data, x_hd_data):
    THRESHOLD = 0.5
    x_data = cv2.resize(x_data, (256,256))
    x = read_image_array(x_data)
    x_hd = read_image_array(x_hd_data)
    _, h, w, _ = x.shape
    # y_pred1 = parse(_model.predict(x)[0][..., -2])
    # y_pred2 = parse(_model.predict(x)[0][..., -1])
    start_time=time.time()
    y_pred3 = parse(_model.predict(x)[0][..., -1])
    time2=time.time()-start_time
    y_pred3[y_pred3 < THRESHOLD] = 0
    y_pred3[y_pred3 > THRESHOLD] = 1
    # src1= cv2.cvtColor(np.asarray(x[0] * 255.0),cv2.COLOR_RGB2BGR)
    src1 = cv2.cvtColor(np.asarray(x_hd_data), cv2.COLOR_RGB2BGR)
    print('src1.shape', src1.shape)
    src2 = cv2.cvtColor(np.asarray(mask_to_3d(y_pred3) * 255.0), cv2.COLOR_RGB2BGR)
    src2 = cv2.resize(src2, org_maskSize, interpolation=cv2.INTER_CUBIC)
    src2 = multiplyGreen(src2)
    x_offset = org_maskCoordinate1[0]
    y_offset = org_maskCoordinate1[1]
    dst = src1
    print('dst.shape', dst.shape)
    print('src2.shape', src2.shape)
    print(src2[0, 0])
    dst[y_offset:y_offset + src2.shape[0], x_offset:x_offset + src2.shape[1]] = src2 * 0.2 + dst[y_offset:y_offset +
                                                                                                          src2.shape[0],
                                                                                             x_offset:x_offset +
                                                                                                      src2.shape[
                                                                                                          1]] * 0.8
    # dst = cv2.addWeighted(src1, 0.8, src2, 0.2, 0)
    return dst,time2


smooth = 1


def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def load_model_weight_newdb(path):
    with CustomObjectScope({
        'dice_loss': dice_loss,
        'dice_coef': dice_coef,
        'bce_dice_loss': bce_dice_loss,
        'focal_loss': focal_loss,
        'iou': iou
        }):
        model = load_model(path)
    return model
    # model = build_model(256)
    # model.load_weights(path)
    # return model


model_seg = load_model_weight_newdb("model.h5")
model_cla = tf.keras.models.load_model("./denseNET169_17_21/018")

#model_cla.summary()


print("Generate predictions for validation data")
prediction_score_for_ROC = []
prediction_label_for_ROC = []
prediction_flattened = []
prediction_true_label = []
time_array = []
time_wSeg_array = []
time_woSeg_array = []
iouHistory = []
print(tf.data.experimental.cardinality(validation_dataset).numpy())

validation_dataset_iterator = validation_dataset.as_numpy_iterator()
mask_dataset_iterator = mask_dataset.as_numpy_iterator()
validation_hd_dataset_iterator = validation_hd_dataset.as_numpy_iterator()

max_batch_num = tf.data.experimental.cardinality(validation_dataset)

ignoreTriggerMode = False
ignoreTrigger = False
ignoreTriggerIndex = 0

for j in range(max_batch_num):
    image_batch, label_batch= validation_dataset_iterator.next()
    image_mask_batch, label_mask_batch = mask_dataset_iterator.next()
    image_hd_batch, label_hd_batch = validation_hd_dataset_iterator.next()
    start = time.time()
    predictions = model_cla.predict_on_batch(image_batch)
    time1=time.time()-start
    prediction_true_label = [*prediction_true_label, *label_batch]
    for i in range(len(predictions)):
        name = output_dir + '{}'.format(validation_dataset.file_paths[j * BATCH_SIZE].split("/")[-1])
        print(name)
        _zeroArray = np.zeros(3)
        _zeroArray[label_batch[i]] = 1
        _addIOU = False
        prediction_label_for_ROC.append(_zeroArray)
        predictions_softmax = tf.nn.softmax(predictions)
        prediction_score_for_ROC.append(predictions_softmax[i])
        print("index:{}, array:{}, max index:{}".format(i + j * BATCH_SIZE + 1, predictions[i],
                                                        np.argmax(predictions[i])))
        prediction_flattened.append(np.argmax(predictions[i]))
        if (np.argmax(predictions[i]) == 0):
            if (np.argmax(predictions[i]) == label_batch[i]):
                print("predict is nerve and true")
                _addIOU = True
                dst, _iouValue,time2 = evaluate_1(model_seg, image_batch[i], image_hd_batch[i], image_mask_batch[i])
            else:
                print("predict is nerve and false")
                _addIOU = True
                dst,time2 = predict_1(model_seg, image_batch[i], image_hd_batch[i])
                _iouValue = 0
        else:
            print("predict is not nerve")
            dst = image_hd_batch[i]
            time2=0
        end = time.time()  # repair the time (Only model inference)
        time_array.append(time1+time2)
        if (np.argmax(predictions[i]) == 0):
            time_wSeg_array.append(end - start)
        else:
            time_woSeg_array.append(end - start)
        print("time used for a pic: ", end - start)
        if class_names[np.argmax(predictions[i])]=="Opening Wound":
          maxWidth = max(cv2.getTextSize("predict: {}".format("General Field"),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1)[0][0]
                       , cv2.getTextSize("time: {}ms".format(round((end - start) * 1000, 3)),
                                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1)[0][0])
        else:
          maxWidth = max(cv2.getTextSize("predict: {}".format("General Field"),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1)[0][0]
                           , cv2.getTextSize("time: {}ms".format(round((end - start) * 1000, 3)),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1)[0][0])
        #print(maxWidth)
        if class_names[np.argmax(predictions[i])] == "Opening Wound":
            cv2.putText(dst, "predict: {}".format("General Field"), (1080 - maxWidth, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(dst, "predict: {}".format(class_names[np.argmax(predictions[i])]), (1080 - maxWidth, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(dst, "time: {}ms".format(round((end - start) * 1000, 3)), (1080 - maxWidth, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 1, cv2.LINE_AA)
        if (_addIOU):
            if (_iouValue > 1):
                _iouValue = 0
            _iouValue = np.around(_iouValue, decimals=5)
            print('iou value:', _iouValue)
            iouHistory.append(_iouValue)
            #cv2.putText(dst, "IOU:" + str(_iouValue), (1080 - maxWidth, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        #(0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(name, dst)
