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
def get_max_for_each_slice(data):
    '''
    data has to be 3d
    '''
    assert len(data.shape) == 3 , 'input data is not 3d'
    max_list = []
    for slice_num in range(data.shape[2]):
        max_list.append(np.max(data[:,:,slice_num]))
    return max_list

def define_laterality(data,threshold):
    midline = int(data.shape[0] / 2)
    max_list_gt = get_max_for_each_slice(data)
    lesion_left = 0
    lesion_right = 0
    lesion_side = ''
    for slice_num in range(len(max_list_gt)):
        if max_list_gt[slice_num] > threshold:
            if np.sum(data[:midline, :, slice_num]) > np.sum(data[midline:, :,
                                                                slice_num]):  ## If stroke in Left side of the image and Right side of the brain
                lesion_left += 1
            elif np.sum(data[:midline, :, slice_num]) < np.sum(data[midline:, :,
                                                                  slice_num]):  ## If stroke in Right side of the image and Left side of the brain
                lesion_right += 1
    if (lesion_left > lesion_right and (lesion_right > 3)) or (lesion_left < lesion_right and (lesion_left > 3)):
        lesion_side = 'B'
    elif lesion_left > lesion_right:
        lesion_side = 'L'
    elif lesion_right > lesion_left:
        lesion_side = 'R'
    # print(lesion_left,lesion_right)
    return lesion_side
list_nonreper_test1 = ["03032","01020","01002","05002","08009","03045","11003","01042","30058A","30082A","08010","30049A","30073A"]
list_nonreper_test2 = ["09005","05011","05010","03002","01021","10002","30008A","30054","01010","30030A","01047","30032","03028"]
list_nonreper_test3 = ["09007","05006","11004","01017","03040","01007","05008","30007","01036","09006","09002","30043","10001"]
list_nonreper_test4 = ["03025","30037","09003","05007","05012","05003","30027A","05005","11002","10006","10007","30084A","02003"]
list_nonreper_test5 = ["01027","01038","01041","02005","01040","08007","10004","03018","09004","01004","01045","10005","11001"]
list_reper_test1 = ["30012","01028","30042","30116","03027","30006A","03017","30068","03043","03016","01003","03046","30099","30117","30046A","30102","03042","30028","03047","01029","01006"]
list_reper_test2 = ["08005","01048","30063","30101","05001","10003","03008","30069A","08008","30035A","03007","30041","30040","30096","30126","08003","30108","30097","03033","30120","01001"]
# 30018A belongs to test 3, but with no GRE. so temperarily removed. if tested in any combination with no GRE, should include 30018A.
list_reper_test3 = ["30018A","03036","30122","30048A","30053","03039","30098","30055","03026","01015","03037","30127","30078A","30071A","03024","01043","03011","30057A","30024A","30115","30002A"]
list_reper_test4 = ["03013","03035","08001","03031","30056","02004","03041","01032","03019","30103","30023","30061","03009","03020","30047A","30026A","30109","05009","03048","10009","30090A"]
list_reper_test5 = ["30092","30022A","30034","30106","30080","03001","01049","12001","30113","01044","30075A","30059A","03003","30077A","30045A","30124","02006","30001A"]
# grouped based on reperfusion rate <30% or >70%
# list_nonreper_test1 = ["30032","01041","30006A","10006","05010","30054","08007"]
# list_nonreper_test2 = ["01020","03040","03008","01017","03001","01040","30058A"]
# list_nonreper_test3 = ["11001","10005","11002","30082A","05002","30027A","03041"]
# list_nonreper_test4 = ["30043","03027","30037","03032","01027","30073A","01021"]
# list_nonreper_test5 = ["11003","01003","10002","09004","05005","30049A","09002"]
# list_reper_test1 = ["30077A","30108","08010","03042","09005","30023","30059A","30045A","08005","03035","03020","30061","03036","10009"]
# list_reper_test2 = ["03048","08008","03033","01036","03037","30101","30022A","05011","30098","08003","30056","05012","30122","05001"]
# list_reper_test3 = ["30041","10004","30028","30047A","09007","02005","05009","30046A","30126","30042","09006","10003","30002A","01010"]
# list_reper_test4 = ["12001","30055","03043","03031","03013","30124","03019","01029","30035A","08001","03047","30127","03028","01015"]
# list_reper_test5 = ["30113","01006","30075A","30001A","03026","30026A","03011","10007","03024","01044","01043","03009","03016","30096"]
subj_list_core = sorted(list_reper_test1)+sorted(list_reper_test2)+sorted(list_reper_test3)+sorted(list_reper_test4)+sorted(list_reper_test5)
subj_list_penumbra = sorted(list_nonreper_test1)+sorted(list_nonreper_test2)+sorted(list_nonreper_test3)+sorted(list_nonreper_test4)+sorted(list_nonreper_test5)
print(subj_list_core)
subj_path = '/Users/admin/deepstroke173/deepstroke173/'

threshold_true = 0.9
# threshold_pred = 60
list_result = {'subject': [], 'auc': [], 'precision': [], 'recall': [], 'specificity': [], 'dice': [], 'auc_all': [],
               'volume_difference': [], 'volume_predicted': [], 'weighted_dice': []}
for subject_name in subj_list_penumbra:
    # for subject_name in ['01007']:
    # load data
    lesion_path = subj_path + subject_name + '/LESION.nii'
    DWI_path = subj_path + subject_name + '/DWI.nii'
    ADC_path = subj_path + subject_name + '/ADC.nii'

    lesion_load = nib.load(lesion_path)
    lesion = lesion_load.get_fdata()
    lesion[np.isnan(lesion)] = 0
    DWI_load = nib.load(DWI_path)
    dwi = DWI_load.get_fdata()
    brain_mask_data = nib.load('/Users/admin/controls_stroke_DL/001/T1.nii')
    brain_mask = brain_mask_data.get_fdata()
    # calcualte dwi mean to remove ventricles.
    dwi = np.maximum(0, np.nan_to_num(dwi, 0))
    dwi = dwi * (brain_mask > 0)
    dwi_mean = np.mean(dwi[np.nonzero(dwi)])
    # load pwi mask
    adc_load = nib.load(ADC_path)
    adc = adc_load.get_fdata()

    # print(lesion[:,:,50])

    y_true_data = []
    y_pred_data = []
    y_pred_continuous_data = []

    if subject_name in ['08005', '08008', '08009']:
        adc_threshold = 310
    else:
        adc_threshold = 620

    lesion_side = define_laterality(lesion[:,:,13:73], threshold_true)

    midline = int(lesion.shape[0] / 2)
    if lesion_side != 'B':
        if lesion_side == 'L':  ## If stroke in Left side of the image and Right side of the brain
            brain_mask[midline:, :, :] = 0
        elif lesion_side == 'R':  ## If stroke in Right side of the image and Left side of the brain
            brain_mask[:midline, :, :] = 0
        else:
            print('check code and data. Left lesion  = Right lesion ')

    for slice_num in range(13,73):
        y_pred_raw = np.logical_and(adc[:, :, slice_num] > 0, adc[:, :, slice_num] < adc_threshold)
        if np.max(lesion[:, :, slice_num]) > threshold_true or np.max(y_pred_raw) > 0.5:
            y_pred_raw = morphology.remove_small_objects(y_pred_raw, 10) ## remove small objects below 1ml
            mask = brain_mask[:, :, slice_num] > 0
            mask = mask * 1.0
            mask[mask == 0] = np.NaN
            dwi_mask = dwi[:, :, slice_num] > (0.3 * dwi_mean)  # remove ventricles
            y_true_masked = (lesion[:, :, slice_num] * mask) * dwi_mask
            y_true_data.append(y_true_masked)
            y_pred_masked = (y_pred_raw * mask) * dwi_mask
            y_pred_data.append(y_pred_masked)
            y_pred_cont = adc[:, :, slice_num] * mask * dwi_mask
            y_pred_continuous_data.append(y_pred_cont)

    y_true = np.array(y_true_data).flatten()
    y_true = y_true[~np.isnan(y_true)]
    y_pred = np.array(y_pred_data).flatten()
    y_pred = y_pred[~np.isnan(y_pred)]
    y_pred_continuous = np.array(y_pred_continuous_data).flatten()
    y_pred_continuous = y_pred_continuous[~np.isnan(y_pred_continuous)]


    fpr, tpr, thresholds = roc_curve(y_true > threshold_true, y_pred_continuous)
    auc_hemisphere = auc(fpr, tpr)
    precision = precision_score(y_true > threshold_true, y_pred > 0.5)
    recall = recall_score(y_true > threshold_true, y_pred > 0.5)
    dice = dice_score(y_true, y_pred, threshold_true=threshold_true, threshold_pred=0.5)
    spec = specificity(y_true, y_pred, threshold_true=threshold_true, threshold_pred=0.5)
    voldiff = vol_diff(y_true, y_pred, threshold_true=threshold_true, threshold_pred=0.5)
    volpred = vol_pred(y_pred, 0.5)
    weighted_dice_score = weighted_dice(y_true, y_pred, threshold_true=threshold_true, threshold_pred=0.5)
    # print(np.sum(y_true>=0.9))
    # print(subject_name, dice, 1-auc_hemisphere, precision, recall, spec, voldiff*0.008, volpred*0.008,abs(voldiff*0.008))
    print(volpred*0.008)
    list_result['subject'].append(subject_name)
    list_result['auc'].append(auc_hemisphere)
    list_result['precision'].append(precision)
    list_result['recall'].append(recall)
    list_result['dice'].append(dice)
    list_result['specificity'].append(spec)
    list_result['volume_difference'].append(voldiff)
    list_result['volume_predicted'].append(volpred)
    list_result['weighted_dice'].append(weighted_dice_score)

# print('mean',np.mean(list_result['dice']),1-np.mean(list_result['auc']),np.mean(list_result['precision']),np.mean(list_result['recall']),np.mean(list_result['specificity']),np.mean(list_result['volume_difference']))
