import nibabel as nib
import glob
import os
from shutil import copyfile
import numpy as np
from nibabel import processing

def glob_filename_list(path,ext=''):
  filename_list = []
  for filename in glob.glob(path+'*'+ext):
    fn = filename.split('/')
    filename_list.append(fn[-1])
  return filename_list
def get_max_for_each_slice(data):
  '''
    data has to be 3d
    '''
  assert len(data.shape) == 3, 'input data is not 3d'
  max_list = []
  for slice_num in range(data.shape[2]):
    max_list.append(np.max(data[:, :, slice_num]))
  return max_list
'''
This file helps mask the inputs images based on MTT map. remove all the information that's not covered by PWI.
modify the contrast list, input path as needed.
'''
inputpath = '/Users/admin/deepstroke173/deepstroke173/'
pt_list = glob_filename_list(inputpath)
contrast_list = ['ADC','DWI','ADCthresholded','LESION','FLAIR','GRE','PWIMTT', 'PWITMAX','PWICBF','PWICBV','TMAXthresholded'] # contrasts that needs mask
copy_list = []
image = {}
for contrast in contrast_list:
  image[contrast] = []
print('dictionary for image storage:',image)
print(pt_list)
for pt in pt_list:
  pt_path = inputpath + pt + '/'
  output_path = '/Users/admin/deepstroke173/PWImasked185/' + pt + '/'

  mtt_load = nib.load(pt_path + 'PWIMTT.nii')
  mtt_data = processing.smooth_image(mtt_load, 4)
  mtt = mtt_data.get_fdata()
  mtt_mask = mtt > 0
  max_mtt = get_max_for_each_slice(mtt_mask)
  start = max_mtt.index(1)
  end = len(max_mtt) - 1 - max_mtt[::-1].index(1)
  for contrast in contrast_list:
    path = inputpath + pt + '/' + contrast + '.nii'
    if os.path.exists(output_path + contrast + '.nii'):
      print(pt,'already have',contrast,'skipping..')
      continue
    if not os.path.exists(path):
      print('contrast not exist:',contrast)
      continue
    if not os.path.exists(output_path):
      os.makedirs(output_path)

    data = nib.load(path)
    image[contrast] = data.get_fdata() * mtt_mask
    image[contrast] = image[contrast][:,:,13:73]
    output_img = nib.Nifti1Image(image[contrast], data.affine)
    nib.save(output_img, os.path.join(output_path, contrast + '.nii'))
  # for copy in copy_list:
    # copy_source = inputpath + pt + '/' + copy + '.nii'
    # copyfile(copy_source, os.path.join(output_path, copy + '.nii'))
  print(pt,'finished processing')
