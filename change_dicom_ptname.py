import pydicom
import glob
import os

def glob_filename_list(path,ext=''):
  filename_list = []
  for filename in glob.glob(path+'*'+ext):
    fn = filename.split('/')
    filename_list.append(fn[-1])
  return filename_list


dicom_path = '/Users/admin/stroke_DL/results/1220_DWI+TMAX+thresholded_nonreper_nestedU/review_nonreper/'
pt_list = glob_filename_list(dicom_path)
print(pt_list)
for subj_id in pt_list:
  path = dicom_path + subj_id + '/' +subj_id + '_000/'
  dcm_list = glob_filename_list(path, '.dcm')
  print(subj_id,len(dcm_list))
  for dcm_file in dcm_list:
    ds = pydicom.dcmread(path+dcm_file)
    ds.PatientName = subj_id
    output_path = '/Users/admin/stroke_DL/results/1220_DWI+TMAX+thresholded_nonreper_nestedU/dicom_nonreper/'+subj_id +'/'
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    pydicom.filewriter.write_file(output_path+dcm_file,ds)