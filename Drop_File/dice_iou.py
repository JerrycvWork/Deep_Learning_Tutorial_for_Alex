import cv2
import glob
import numpy as np
import scipy.stats

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
def compute_iou(predict_mask, gt_mask):

    if np.sum(predict_mask) == 0 or np.sum(gt_mask) == 0:
        iou_ = 0
        return iou_

    n_ii = np.sum(np.logical_and(predict_mask, gt_mask))
    t_i = np.sum(gt_mask)
    n_ij = np.sum(predict_mask)

    iou_ = n_ii / (t_i + n_ij - n_ii)

    return iou_


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

total_iou=[]
total_dice=[]



counter=0

for filename in glob.glob(r"./val/mask/*"):
    gt=cv2.imread(filename)
    pred = cv2.imread(r"./Results/Nerve Segmentation/20220426/result/val/ori_unet/"+filename.split("/")[-1])


    pred=cv2.resize(pred,(gt.shape[1],gt.shape[0]))

    print(filename)

    gt = gt.astype(np.float32) / 255
    pred = pred.astype(np.float32) / 255

    gt[gt>0.5]=1
    gt[gt < 0.5] = 0
    pred[pred > 0.5] = 1
    pred[pred < 0.5] = 0


    total_dice.append(dice_coef(pred, gt))
    total_iou.append(compute_iou(pred, gt))
    counter += 1

print("Pred 95%: ")
print(mean_confidence_interval(total_iou))
print(mean_confidence_interval(total_dice))

print("Mean std")
print(np.mean(np.asarray(total_iou)),np.std(np.asarray(total_iou)))
print(np.mean(np.asarray(total_dice)),np.std(np.asarray(total_dice)))