import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

''' basic dependencies '''
import numpy as np
import os

from skimage import morphology
import nibabel as nib

'''
contrast
'''

list_contrast_keyword = ['ADC']
output_name = 'ADCthresholded.nii'

ext_data = 'nii'
# to export only part of the image.

'''
get files
'''
# get all file list
dir_stroke = '/Users/admin/deepstroke173/deepstroke173'
dir_preprocessing = '/Users/admin/deepstroke173/deepstroke173'
dir_mask = '/Users/admin/controls_stroke_DL/001/T1.nii'
dir_dwi = '/Users/admin/deepstroke173/deepstroke173'
# list_subfolder = sorted([os.path.join(dir_stroke,x) for x in os.listdir(dir_stroke) if
#                   os.path.isdir(os.path.join(dir_stroke,x))])

subject_cases = ['30089']
list_subfolder = ['/Users/admin/deepstroke173/deepstroke173/' + x for x in subject_cases]
print(list_subfolder)
# loop for subjects
def formmask(list_subfolder,dir_preprocessing,dir_mask, dir_dwi, list_contrast_keyword, output_name, ext_data = 'nii',mode = 'adc',cluster = 10):
    '''

    :param list_subfolder: where to read images
    :param dir_preprocessing: where to save images
    :param dir_mask: brain mask dir
    :param list_contrast_keyword: name for nii files
    :param output_name:
    :param ext_data: nii
    :param mode:
    :return:
    '''
    num_subject = len(list_subfolder)
    for subfolder in list_subfolder:
        subject_name = subfolder.split('/')[-1]
        # output folder
        dir_output = os.path.join(dir_preprocessing, subject_name)
        if not os.path.exists(dir_output):
            os.mkdir(dir_output)

        list_subfiles = os.listdir(subfolder)
        ## YUAN modified on Sep 17th 2018
        # added in an extra criteria to ignore hidden files
        list_files_with_ext = sorted([x for x in list_subfiles if (x.endswith(ext_data) and (not x.startswith('.')))])

        # get each contrast
        list_filename_sample_contrasts = []
        for index_contrast in range(len(list_contrast_keyword)):
            keyword_contrast = list_contrast_keyword[index_contrast]
            list_filename_sample_contrasts.append(
                sorted([x for x in list_files_with_ext if x.find(keyword_contrast)>=0])
                )

        # check number of nifti file vs contrast requirement
        print(len(list_files_with_ext), len(list_contrast_keyword))
        if min([len(x) for x in list_filename_sample_contrasts])<=0 or len(list_files_with_ext) <= len(list_contrast_keyword):
            print('missing contrasts at {0}:'.format(subject_name), list_files_with_ext)
            continue
        # T1 mask
        mask_load = nib.load(dir_mask)
        mask = mask_load.get_fdata()
        mask = mask > 0 * 1.
        dwi_load = nib.load(dir_dwi + '/' + subject_name + '/DWI.nii')
        dwi = dwi_load.get_fdata() * mask
        mean_dwi = np.mean(dwi[np.nonzero(dwi)])
        dwi_mask = dwi > (0.4 * mean_dwi)
        print(mean_dwi*0.4,np.mean(dwi_mask))

        filename_contrast = list_filename_sample_contrasts[0][0]
        img_load = nib.load(subfolder+'/'+filename_contrast)
        img = img_load.get_fdata()
        img = np.maximum(0, np.nan_to_num(img, 0))
        if mode == 'tmax':
            if int(subject_name[:5]) >= 30059:
                img_threshold = 6
                print(subject_name, '>=30059, Tmax threshold = ', img_threshold)
                img_thresholded = (img > img_threshold) * img * 10. * mask * dwi_mask
            else:
                img_threshold = 60
                print(subject_name, '<30059, Tmax threshold = ', img_threshold)
                img_thresholded = (img > img_threshold) * img * 1. * mask * dwi_mask
        elif mode == 'adc':

            if subject_name in ['08005', '08008', '08009']:
                img_threshold = 310
                print(subject_name, 'adc threshold', img_threshold)
            else:
                img_threshold = 620
                print(subject_name, 'adc threshold', img_threshold)
            img_thresholded = np.logical_and(img < img_threshold, img > 0)  * mask
            img_thresholded = (morphology.remove_small_objects(img_thresholded, cluster)) * 1.0

        #export
        print(img_thresholded.shape)
        print(os.path.join(subfolder,output_name))
        TMAXmask = nib.Nifti1Image(img_thresholded, img_load.affine)
        nib.save(TMAXmask, os.path.join(subfolder, output_name))

formmask(list_subfolder,dir_preprocessing,dir_mask, dir_dwi, list_contrast_keyword, output_name, ext_data = 'nii',mode = 'adc',cluster = 10)