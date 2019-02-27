import nibabel as nib
import os

T1path = '/Users/admin/controls_stroke_DL/001/T1.nii'
T1_load = nib.load(T1path)
T1 = T1_load.get_fdata()
T1 = T1[:,:,13:73]
output_img = nib.Nifti1Image(T1, T1_load.affine)
nib.save(output_img, os.path.join('/Users/admin/controls_stroke_DL/001/', 'T1_cerebrum.nii'))