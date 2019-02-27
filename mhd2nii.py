import nibabel as nib
import os
import numpy as np
import h5py
from scipy import ndimage, misc
import SimpleITK as sitk
import numpy as np
import glob

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    return numpyImage

def glob_filename_list(path,ext=''):
    filename_list = []
    for filename in glob.glob(path+'*'+ext):
        fn = filename.split('/')
        filename_list.append(fn[-1])
    return filename_list

filepath = "/Users/admin/D2_RAPID/RAPIDsegmentation/"
outputpath = '/Users/admin/D2_RAPID/RAPIDsegmentation_sorted/'
name_list = glob_filename_list(filepath)

for name in ['05005']:
    ext = '.mhd'
    filename = ['segm_mask_view0_Thresholded_ADC_Parameter_View_slab0','segm_mask_view0_Thresholded_Tmax_Parameter_View_slab0',
                'segm_mask_view1_Thresholded_Tmax_Parameter_View_slab0','segm_mask_view2_Thresholded_ADC_Parameter_View_slab0']

    baseline_path = filepath + name + '/results/'
    baseline_namelist = glob_filename_list(baseline_path,'.nii')
    bl_path = filepath + name + '/baseline_slab0' +ext
    b0_path = filepath + name + '/b0InPWISpace' +ext
    if baseline_namelist:
        baseline_name = baseline_namelist[-2] # find baseline image. [-1] is the AIFVOF
        adc_name = baseline_namelist[0]
        print(name, 'baseline image:', baseline_name)
    else:
        print(name, 'does not have nifty file, please check. skipping...')
        continue

    if not os.path.exists(outputpath + name):
        os.makedirs(outputpath + name)
    for seg_mask in filename:
        img_path = filepath + name + '/' + seg_mask + ext
        if not os.path.exists(img_path):
            print(name,'does not have file:', seg_mask)
            continue
        img_m = load_itk_image(img_path) # np.Array
        img_m = img_m.transpose((2, 1, 0))
        img_m = img_m[:,::-1,...] # transpose and flip to correct direction
        pwi = nib.load(baseline_path + baseline_name)
        img = nib.Nifti1Image(img_m, pwi.affine)
        nib.save(img, outputpath+name +'/' +seg_mask+'.nii')
    bl = load_itk_image(bl_path)
    bl = bl.transpose((2, 1, 0))
    bl = bl[:, ::-1, ...]  # transpose and flip to correct direction
    baseline = nib.Nifti1Image(bl,pwi.affine)
    nib.save(baseline, outputpath + name + '/' + 'baseline.nii')
    b0 = load_itk_image(b0_path)
    b0 = b0.transpose((2, 1, 0))
    b0 = b0[:, ::-1, ...]  # transpose and flip to correct direction
    b0_m = nib.Nifti1Image(b0,pwi.affine)
    nib.save(b0_m, outputpath + name + '/' + 'b0.nii')
    adc = nib.load(baseline_path + adc_name)
    nib.save(adc, outputpath + name +'/adc.nii' )

# filepath = "/Users/admin/D2_RAPID/RAPIDsegmentation/01001/post_perf_allTp_slab0_slice00.mhd"
#
# img_m = load_itk_image(filepath) # np.Array
# img = nib.Nifti1Image(img_m, np.eye(4))
# nib.save(img, 'img.nii')