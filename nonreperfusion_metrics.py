import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_score, recall_score, roc_curve
from nibabel import processing
from skimage import morphology

def dice_score(y_true, y_pred, smooth=0.0000001, threshold_true=0.1, threshold_pred=0.5, model='>'):
    '''

    :param y_true: array for ground truth
    :param y_pred: array for output
    :param smooth: usually no need to change, prevent 0/0 .
    :param threshold_true: above which ground truth is considered 1, the rest is 0
    :param threshold_pred: above / below which output is considered 1, the rest is 0
    :param model: '>' above the threshold_pred; '<' below the threshold_pred but >0
    :return: a dice score.
    '''
    y_true_f = y_true.flatten() >= threshold_true
    if model == '<':
        y_pred_f = np.logical_and(y_pred.flatten() <= threshold_pred, y_pred.flatten() > 0)
    else:
        y_pred_f = y_pred.flatten() >= threshold_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def specificity(y_true, y_pred, smooth=0.00001, threshold_true=0.1, threshold_pred=0.5, model='>'):
    y_neg_f = y_true.flatten() < threshold_true
    if model == '<':
        y_pred_pos_f = np.logical_and(y_pred.flatten() <= threshold_pred, y_pred.flatten() > 0)
    else:
        y_pred_pos_f = y_pred.flatten() >= threshold_pred
    false_pos = np.sum(y_neg_f * y_pred_pos_f)
    return np.sum(y_neg_f) / (np.sum(y_neg_f) + false_pos + smooth)


def vol_diff(y_true, y_pred, threshold_true=0.1, threshold_pred=0.5, model='>'):
    y_true_f = y_true.flatten() >= threshold_true
    if model == '<':
        y_pred_f = np.logical_and(y_pred.flatten() <= threshold_pred, y_pred.flatten() > 0)
    else:
        y_pred_f = y_pred.flatten() >= threshold_pred
    return np.sum(y_pred_f) - np.sum(y_true_f)

def vol_pred(y_pred,threshold_pred=0.5):
    y_pred_f = y_pred.flatten() >= threshold_pred
    return np.sum(y_pred_f)

def weighted_dice(y_true,y_pred,smooth = 0.00001,threshold_true = 0.1, threshold_pred =0.5):
    y_true_f = y_true.flatten() >= threshold_true
    y_pred_f = y_pred.flatten() >= threshold_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (1.7 * np.sum(y_true_f) + 0.3 * np.sum(y_pred_f) + smooth)
# list_nonreper_test1 = ["03032","01020","01002","05002","08009","03045","11003","01042","30058A","30082A","08010","30049A","30073A"]
# list_nonreper_test2 = ["09005","05011","05010","03002","01021","10002","30008A","30054","01010","30030A","01047","30032","03028"]
# list_nonreper_test3 = ["09007","05006","11004","01017","03040","01007","05008","30007","01036","09006","09002","30043","10001"]
# list_nonreper_test4 = ["03025","30037","09003","05007","05012","05003","30027A","05005","11002","10006","10007","30084A","02003"]
# list_nonreper_test5 = ["01027","01038","01041","02005","01040","08007","10004","03018","09004","01004","01045"]
# grouped based on reperfusion rate <30% or >70%
list_nonreper_test1 = ["30032","01041","30006A","10006","05010","30054","08007"]
list_nonreper_test2 = ["01020","03040","03008","01017","03001","01040","30058A"]
list_nonreper_test3 = ["11001","10005","11002","30082A","05002","30027A","03041"]
list_nonreper_test4 = ["30043","03027","30037","03032","01027","30073A","01021"]
list_nonreper_test5 = ["11003","01003","10002","09004","05005","30049A","09002"]
list_reper_test1 = ["30077A","30108","08010","03042","09005","30023","30059A","30045A","08005","03035","03020","30061","03036","10009"]
list_reper_test2 = ["03048","08008","03033","01036","03037","30101","30022A","05011","30098","08003","30056","05012","30122","05001"]
list_reper_test3 = ["30041","10004","30028","30047A","09007","02005","05009","30046A","30126","30042","09006","10003","30002A","01010"]
list_reper_test4 = ["12001","30055","03043","03031","03013","30124","03019","01029","30035A","08001","03047","30127","03028","01015"]
list_reper_test5 = ["30113","01006","30075A","30001A","03026","30026A","03011","10007","03024","01044","01043","03009","03016","30096"]
subj_list_core = sorted(list_reper_test1)+sorted(list_reper_test2)+sorted(list_reper_test3)+sorted(list_reper_test4)+sorted(list_reper_test5)
subj_list_penumbra = sorted(list_nonreper_test1)+sorted(list_nonreper_test2)+sorted(list_nonreper_test3)+sorted(list_nonreper_test4)+sorted(list_nonreper_test5)
print(subj_list_core)
subj_path = '/Users/admin/deepstroke173/deepstroke173/'

threshold_true = 0.9
# threshold_pred = 60
list_result = {'auc': [], 'precision': [], 'recall': [], 'dice': [], 'auc_all': []}
for subject_name in subj_list_penumbra:
    # for subject_name in ['01007']:
    # load data
    TMAX_path = subj_path + subject_name + '/PWITMAX.nii'
    lesion_path = subj_path + subject_name + '/LESION.nii'
    DWI_path = subj_path + subject_name + '/DWI.nii'
    ADC_path = subj_path + subject_name + '/ADC.nii'
    TMAX_load = nib.load(TMAX_path)
    ## for tmax map, smooth out the image
    tmax_smooth = processing.smooth_image(TMAX_load, 3)
    tmax = tmax_smooth.get_fdata()

    lesion_load = nib.load(lesion_path)
    lesion = lesion_load.get_fdata()
    lesion[np.isnan(lesion)] = 0
    DWI_load = nib.load(DWI_path)
    dwi = DWI_load.get_fdata()
    brain_mask_data = nib.load('/Users/admin/controls_stroke_DL/001/T1.nii')
    brain_mask = brain_mask_data.get_fdata()
    # calcualte dwi mean to remove ventricles.
    mask_fordwi = brain_mask[:, :, 45] > 0
    mask_dwi = mask_fordwi * 1.0
    mask_dwi[mask_dwi == 0] = np.NaN
    dwi_masked = dwi[:, :, 45] * mask_dwi
    dwi_num = dwi_masked.flatten()
    dwi_mean = np.mean(dwi_num[~np.isnan(dwi_num)])
    # load pwi mask
    adc_load = nib.load(ADC_path)
    adc = adc_load.get_fdata()

    # print(lesion[:,:,50])

    y_true_data = []
    y_pred_data = []

    if subject_name in ['30073A', '30082A', '30084A']:
        # print(subject_name)
        threshold_pred = 6
    else:
        threshold_pred = 60

    if subject_name in ['08005', '08008', '08009']:
        adc_threshold = 310
    else:
        adc_threshold = 620

    lesion_side = ''
    for slice_num in range(13,73):
        if np.max(lesion[:, :, slice_num]) > threshold_true:
            if np.sum(lesion[0:46, :, slice_num]) > np.sum(
                    lesion[46:91, :, slice_num]):  ## If stroke in Left side of the image and Right side of the brain
                lesion_side += 'L'
            elif np.sum(lesion[0:46, :, slice_num]) < np.sum(
                    lesion[46:91, :, slice_num]):  ## If stroke in Right side of the image and Left side of the brain
                lesion_side += 'R'
        else:
            continue
    if (lesion_side.count('L') > lesion_side.count('R') and (lesion_side.count('R') > 1)) or (
            lesion_side.count('L') < lesion_side.count('R') and (lesion_side.count('L') > 1)):
        lesion_side = 'B'
        # print('bilateral stroke',lesion_side,'L:',lesion_side.count('L'),'R:',lesion_side.count('R'))
    for slice_num in range(13,73):
        if np.max(lesion[:,:,slice_num])> threshold_true or np.max(tmax[:,:,slice_num])>threshold_pred:
        # if np.max(lesion[:, :, slice_num]) > threshold_true:
            y_pred_raw = np.logical_or(tmax[:, :, slice_num] > threshold_pred, adc[:, :, slice_num] < adc_threshold)
            y_pred_raw = morphology.remove_small_objects(y_pred_raw, 125) ## remove small objects below 100 pixel
            mask = brain_mask[:, :, slice_num] > 0
            mask = mask * 1.0
            # select out the hemisphere with stroke lesions
            if lesion_side == 'B':
                mask = mask
            elif lesion_side.count('L') > lesion_side.count(
                    'R'):  ## If stroke in Left side of the image and Right side of the brain
                mask[46:91, :] = np.NaN

            elif lesion_side.count('L') < lesion_side.count(
                    'R'):  ## If stroke in Right side of the image and Left side of the brain
                mask[0:46, :] = np.NaN

            else:
                print('lesion side cannot decide, check data and code')

            mask[mask == 0] = np.NaN
            dwi_mask = dwi[:, :, slice_num] > (0.3 * dwi_mean)  # remove ventricles
            y_true_masked = (lesion[:, :, slice_num] * mask) * dwi_mask
            y_true_data.append(y_true_masked)
            y_pred_masked = (y_pred_raw * mask) * dwi_mask
            y_pred_data.append(y_pred_masked)
        else:
            continue
    y_true = np.array(y_true_data).flatten()
    y_true = y_true[~np.isnan(y_true)]
    y_pred = np.array(y_pred_data).flatten()
    y_pred = y_pred[~np.isnan(y_pred)]


    fpr, tpr, thresholds = roc_curve(y_true > threshold_true, y_pred)
    auc_hemisphere = auc(fpr, tpr)
    precision = precision_score(y_true > threshold_true, y_pred > 0.5)
    recall = recall_score(y_true > threshold_true, y_pred > 0.5)
    dice = dice_score(y_true, y_pred, threshold_true=threshold_true, threshold_pred=0.5)
    spec = specificity(y_true, y_pred, threshold_true=threshold_true, threshold_pred=0.5)
    voldiff = vol_diff(y_true, y_pred, threshold_true=threshold_true, threshold_pred=0.5)
    volpred = vol_pred(y_pred, 0.5)
    weighted_dice_score = weighted_dice(y_true, y_pred, threshold_true=threshold_true, threshold_pred=0.5)
    # print(np.sum(y_true>=0.9))
    print(subject_name, dice, auc_hemisphere, precision, recall, spec, voldiff*0.008, volpred*0.008, weighted_dice_score)
