import nibabel as nib
import numpy as np


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
    print(lesion_left,lesion_right)
    return lesion_side
alpha = 0.5 # color hue
thres = 0.2
thres_gt = 0.9

# model_name ='1129_charlesmod_DWI+ADC_reper_aug_true0.9_nonan'
model_name = '1219_DWI+TMAX+thresholded_nonreper'
name_tag = '_if_nonreper_dwitmax'
data_dir = '/Users/admin/stroke_DL/results/' + model_name +'/' # On longo:'/data3/yuanxie/project_stroke_mask/'
subj_id = '03032'
gt_path = data_dir + 'gt_' + model_name + '_fold1_' + subj_id + '.nii'
flair_path = data_dir + 'flair_' + model_name + '_fold1_' + subj_id + '.nii'
gt = nib.load(gt_path)
gt_volume = gt.get_fdata()
# max_list = get_max_for_each_slice(gt_volume)
# print(len(max_list))
# print(max_list)
lesion_side = define_laterality(gt_volume,thres_gt)
print(lesion_side)