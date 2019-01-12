import pydicom
import glob
import os

def glob_filename_list(path,ext=''):
  filename_list = []
  for filename in glob.glob(path+'*'+ext):
    fn = filename.split('/')
    filename_list.append(fn[-1])
  return filename_list

model_name = '1219_DWI+TMAX+thresholded_nonreper'
dicom_path = '/Users/admin/stroke_DL/results/'+model_name + '/review_nonreper_thres0.3/'
pt_list = glob_filename_list(dicom_path)
print(pt_list)
for subj_id in pt_list:
  path = dicom_path + subj_id + '/' +subj_id + '_000/'
  dcm_list = glob_filename_list(path, '.dcm')
  print(subj_id,len(dcm_list))
  for dcm_file in dcm_list:
    ds = pydicom.dcmread(path+dcm_file)
    ds.PatientName = subj_id
    ds.PatientID = subj_id
    ds.StudyDescription = model_name
    output_path = '/Users/admin/stroke_DL/results/'+model_name + '/dicom_nonreper/'+subj_id +'/'
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    pydicom.filewriter.write_file(output_path+dcm_file,ds)