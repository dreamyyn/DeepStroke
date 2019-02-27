import sys
sys.path.insert(0, '/Users/admin/stroke_DL/script_stroke')
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_score, recall_score, roc_curve
from nibabel import processing
from skimage import morphology
from create_fig_for_model import *

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

def metrics_output(y_true, y_pred, threshold_true, threshold_pred):
  '''output all the metrics including auc dice recall precision f1score and volume difference'''
  fpr, tpr, thresholds = roc_curve(y_true > threshold_true, y_pred)
  auc_hemisphere = auc(fpr, tpr)
  precision = precision_score(y_true > threshold_true, y_pred > threshold_pred)
  recall = recall_score(y_true > threshold_true, y_pred > threshold_pred)
  dice = dice_score(y_true, y_pred, threshold_true=threshold_true, threshold_pred=threshold_pred)
  spec = specificity(y_true, y_pred, threshold_true=threshold_true, threshold_pred=threshold_pred)
  voldiff = 0.008 * vol_diff(y_true, y_pred, threshold_true=threshold_true, threshold_pred=threshold_pred)
  volpred = 0.008 * vol_pred(y_pred, threshold_pred)
  f1score = 2 * precision * recall / (precision + recall + 0.0001)
  return auc_hemisphere, precision, recall, dice, spec, voldiff, volpred, f1score, fpr, tpr

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
#list for testing, grouped based on TICI >=2b or <=2a
list_nonreper_test1 = ["03032","01020","01002","05002","08009","03045","11003","01042","30058A","30082A","08010","30049A","30073A","30019"]
list_nonreper_test2 = ["09005","05011","05010","03002","01021","10002","30008A","30054","01010","30030A","01047","30032","03028","30087"]
list_nonreper_test3 = ["09007","05006","11004","01017","03040","01007","05008","30007","01036","09006","09002","30043","10001"]
list_nonreper_test4 = ["03025","30037","09003","05007","05012","05003","30027A","05005","11002","10006","10007","30084A","02003"]
list_nonreper_test5 = ["01027","01038","01041","02005","01040","08007","10004","03018","09004","01004","01045","10005","11001"]

#remove 01028 (1), 03003 (5), 30022 (5)
# list_reper_test1 = ["30012","30042","30116","03027","30006A","03017","30068","03043","03016","01003","03046","30099","30117","30046A","30102","03042","30028","03047","01029","01006","30044"]
# list_reper_test2 = ["08005","01048","30063","30101","05001","10003","03008","30069A","08008","30035A","03007","30041","30040","30096","30126","08003","30108","30097","03033","30120","01001","30039"]
# # 30018A belongs to test 3, but with no GRE. so temperarily removed. if tested in any combination with no GRE, should include 30018A. 30071 replaced by 30105
# list_reper_test3 = ["30018A","03036","30122","30048A","30053","03039","30098","30055","03026","01015","03037","30127","30078A","30105","03024","01043","03011","30057A","30024A","30115","30002A","30016"]
# list_reper_test4 = ["03013","03035","08001","03031","30056","02004","03041","01032","03019","30103","30023","30061","03009","03020","30047A","30026A","30109","05009","03048","10009","30090A","30015"]
# list_reper_test5 = ["30092","30034","30106","30080","03001","01049","12001","30113","01044","30075A","30059A","30077A","30045A","30124","02006","30001A","30005","30011","30014","30051","01023"]

# 08004 removed from test 2 because no lesion; remove 30071 because no Tmax lesion
#remove 01028,03003,30022
list_all_test1 = ["03043","30034","30124","30037","30069A","30058A","01027","30023","12001","03045","10002","08007","30109","05011","10009","01047","10005","30097","11001","10001","08009","30095","30117","11002","30113","03046","09005","30082A","30007","03007","30032","03048","01017","03019","30005","30011"]
list_all_test2 = ["03026","01001","05006","05008","30102","01049","01015","30080","30061","30045A","30054","08010","30068","03036","11003","09007","01042","10006","30012","30053","03042","09004","30063","01003","01038","30110","08001","02005","30101","30006A","01006","30127","03039","05007","30014","30015"]
list_all_test3 = ["03011","30057A","01040","03008","30024A","01002","03032","30042","03028","30108","05010","30104","03001","30084A","03025","30075A","30106","05002","30027A","01010","30103","30047A","01007","03047","05009","03035","30049A","03018","30115","30028","03041","30002A","03024","01045","30016","30019"]
list_all_test4 = ["03009","01041","03037","30090A","02004","30041","01020","30025","02001","30040","30122","30077A","09002","30120","03013","02003","05003","30098","01021","08008","11004","30008A","05012","03033","01048","30099","30105","01044","30030A","30048A","30055","30018A","03020","03031","30039","30044"]
list_all_test5 = ["03017","30035A","30001A","01043","30096","01004","10003","08005","30073A","30056","05001","03027","09006","30100","08003","10004","30116","30092","02006","30026A","30046A","03002","03040","09003","30059A","30043","01029","03016","30078A","10007","05005","01036","30126","01032","30051","30087","30089","01023"]
# grouped based on reperfusion rate <30% or >70%
# list_nonreper_test1 = ["30032","01041","30006A","10006","05010","30054","08007"]
# list_nonreper_test2 = ["01020","03040","03008","01017","03001","01040","30058A"]
# list_nonreper_test3 = ["11001","10005","11002","30082A","05002","30027A","03041"]
# list_nonreper_test4 = ["30043","03027","30037","03032","01027","30073A","01021"]
# list_nonreper_test5 = ["11003","01003","10002","09004","05005","30049A","09002"]
list_reper_test1 = ["30077A","30108","08010","03042","09005","30023","30059A","30045A","08005","03035","03020","30061","03036","10009","30005","30011"]
list_reper_test2 = ["03048","08008","03033","01036","03037","30101","05011","30098","08003","30056","05012","30122","05001","30014","30015"]
list_reper_test3 = ["30041","10004","30028","30047A","09007","02005","05009","30046A","30126","30042","09006","10003","30002A","01010","30016"]
list_reper_test4 = ["12001","30055","03043","03031","03013","30124","03019","01029","30035A","08001","03047","30127","03028","01015","30087"]
list_reper_test5 = ["30113","01006","30075A","30001A","03026","30026A","03011","10007","03024","01044","01043","03009","03016","30096","30039"]
subj_list_core = sorted(list_reper_test1)+sorted(list_reper_test2)+sorted(list_reper_test3)+sorted(list_reper_test4)+sorted(list_reper_test5)
subj_list_penumbra = sorted(list_nonreper_test1)+sorted(list_nonreper_test2)+sorted(list_nonreper_test3)+sorted(list_nonreper_test4)+sorted(list_nonreper_test5)
subj_list_all = list_all_test1 + list_all_test2 + list_all_test3 + list_all_test4 + list_all_test5
print(subj_list_core)
subj_path = '/Users/admin/deepstroke173/PWImasked185/'
RAPID_path = '/Users/admin/D2_RAPID/segmentation/'
testmode = 'Tmax1' #"Tmax","Tmax1" "ADC"
threshold_true = 0.9
# threshold_pred = 60
list_result = {'subject': [], 'auc': [], 'precision': [], 'recall': [], 'specificity': [], 'dice': [],
               'volume_difference': [], 'volume_predicted': [],'abs_volume_difference':[]}
all_y_true = np.array([])
all_y_pred = np.array([])
all_y_continuous = np.array([])
for subject_name in subj_list_penumbra:
    # for subject_name in ['01007']:
    # load data
    lesion_path = subj_path + subject_name + '/LESION.nii'
    DWI_path = subj_path + subject_name + '/DWI.nii'
    ADC_path = subj_path + subject_name + '/' + testmode +'_seg.nii'
    if not os.path.exists(ADC_path):
        print(subject_name,'have no segmentation data.')
        continue

    lesion_load = nib.load(lesion_path)
    lesion = lesion_load.get_fdata()
    lesion[np.isnan(lesion)] = 0
    DWI_load = nib.load(DWI_path)
    dwi = DWI_load.get_fdata()
    brain_mask_data = nib.load('/Users/admin/controls_stroke_DL/001/T1_cerebrum.nii')
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


    lesion_side = define_laterality(lesion[:,:,:], threshold_true)

    midline = int(lesion.shape[0] / 2)
    if lesion_side != 'B':
        if lesion_side == 'L':  ## If stroke in Left side of the image and Right side of the brain
            brain_mask[midline:, :, :] = 0
        elif lesion_side == 'R':  ## If stroke in Right side of the image and Left side of the brain
            brain_mask[:midline, :, :] = 0
        else:
            print('check code and data. Left lesion  = Right lesion ')

    for slice_num in range(lesion.shape[2]):
        y_pred_raw = (adc[:,:,slice_num] > 0.5) + 0.
        # if np.max(lesion[:, :, slice_num]) > threshold_true or np.max(y_pred_raw) > 0.5:
        # y_pred_raw = morphology.remove_small_objects(y_pred_raw, 10) ## remove small objects below 1ml
        mask = brain_mask[:, :, slice_num] > 0
        mask = mask * 1.0
        mask[mask == 0] = np.NaN
        dwi_mask = dwi[:, :, slice_num] > (0.3 * dwi_mean)  # remove ventricles
        y_true_masked = (lesion[:, :, slice_num] * mask) * dwi_mask
        y_true_data.append(y_true_masked)
        y_pred_masked = (y_pred_raw * mask) * dwi_mask
        y_pred_data.append(y_pred_masked)
        if testmode == 'Tmax1':
            y_pred_continuous_data.append(adc[:,:,slice_num] * mask * dwi_mask)


    y_true = np.array(y_true_data).flatten()
    y_true = y_true[~np.isnan(y_true)]
    y_pred = np.array(y_pred_data).flatten()
    y_pred = y_pred[~np.isnan(y_pred)]
    if testmode == 'Tmax1':
        y_pred_continuous = np.array(y_pred_continuous_data).flatten()
        y_pred_continuous = y_pred_continuous[~np.isnan(y_pred_continuous)]
        all_y_true = np.append(all_y_true, y_true)
        # all_y_pred = np.append(all_y_pred,y_pred)
        all_y_continuous = np.append(all_y_continuous, y_pred_continuous)

    auc_hemisphere, precision, recall, dice, spec, voldiff, volpred, f1score, fpr1, tpr1 = metrics_output(y_true, y_pred,
                                                                                                    threshold_true,
                                                                                                    0.5)
    # print(np.sum(y_true>=0.9))
    print(subject_name, dice, auc_hemisphere, precision, recall, spec, voldiff, volpred,abs(voldiff))
    # print(volpred*0.008)
    list_result['subject'].append(subject_name)
    list_result['auc'].append(auc_hemisphere)
    list_result['precision'].append(precision)
    list_result['recall'].append(recall)
    list_result['dice'].append(dice)
    list_result['specificity'].append(spec)
    list_result['volume_difference'].append(voldiff)
    list_result['volume_predicted'].append(volpred)
    list_result['abs_volume_difference'].append(abs(voldiff))
save_dict(list_result,RAPID_path,filename= testmode + '_RAPID_metrics.csv')
if testmode == 'Tmax1':
    fpr, tpr, thresholds = roc_curve(all_y_true > threshold_true, all_y_continuous, pos_label=1)
    roc_auc = auc(fpr, tpr)
    create_roc(fpr,tpr,roc_auc,RAPID_path, thresholds, figname=testmode + '_roc.png',tablename = testmode+'.csv')
# print('mean',np.mean(list_result['dice']),1-np.mean(list_result['auc']),np.mean(list_result['precision']),np.mean(list_result['recall']),np.mean(list_result['specificity']),np.mean(list_result['volume_difference']))
