import pydicom
import glob
import os
import csv
import pprint

def glob_filename_list(path,ext=''):
  filename_list = []
  for filename in glob.glob(path+'*'+ext):
    fn = filename.split('/')
    filename_list.append(fn[-1])
  return filename_list


# with open("ZJU_anonymize_list.csv", mode='r', encoding='utf-8-sig') as infile: # csv with the format of (id,name) per line.
#   reader = csv.reader(infile)
#   with open('ZJU_new.csv', mode='w') as outfile:
#     writer = csv.writer(outfile)
#     anonymize_list = {rows[1]: rows[0] for rows in reader} # so that the key is the patient name, and value is the anonymized id.
# print(anonymize_list)

# modify path
dicom_path = '/Users/admin/UCLAdata_RAPID/'
pt_list = glob_filename_list(dicom_path)
# print(pt_list)
for subj_id in pt_list:
  path = dicom_path + subj_id + '/'
  date_list = glob_filename_list(path)
  # if subj_id not in anonymize_list:
  #   print('patient', subj_id, 'not in the anonymization list. please check.')
  #   continue
  dirlist = glob.glob(path + '/*/*/*.dcm')
  for dir in dirlist:
  # anonymize field
    ds = pydicom.dcmread(dir)
    # if ds.PatientName == anonymize_list[subj_id]:
    #   print(subj_id,"already processed. skipping...")
    #   continue
    # if ds.InstitutionName == 'Anonymized':
    #   print(subj_id,"already processed. skipping...")
    #   continue
    if ds.PatientName == ds.PatientID:
      print(subj_id,"already processed. skipping...")
      continue
    ds.PatientName = ds.PatientID
    # ds.PatientID = anonymize_list[subj_id]
    # ds.InstitutionName = 'Anonymized'
    # ds.InstitutionAddress = ''
    # ds.StationName = ''
    # ds.OperatorsName = ''
    # ds.PatientsBirthDate = ds.PatientsBirthDate[:4] + "0101"
    # print(subj_id, ds.PatientName, ds.PatientID, ds.InstitutionName)
    # output_path = dcm_path
    # if not os.path.exists(output_path):
    #   os.makedirs(output_path)
    pydicom.filewriter.write_file(dir,ds) # write out anonymized file.
  print('successfully anonymized',subj_id, 'anonymized patient name, id, and institution:',ds.PatientName, ds.PatientID, ds.InstitutionName, ds.InstitutionAddress)
  # for date in date_list:
  #   scan_path = path + date + '/'
  #   scan_list = glob_filename_list(scan_path)
  #   for scan in scan_list:
  #     dcm_path = scan_path + scan +'/'
  #     dcm_list = glob_filename_list(dcm_path, '.dcm')
  #     # print(subj_id,len(dcm_list))
  #     for dcm_file in dcm_list:
  #       # anonymize field
  #       ds = pydicom.dcmread(dcm_path+dcm_file)
  #       ds.PatientName = anonymize_list[subj_id]
  #       ds.PatientID = anonymize_list[subj_id]
  #       ds.InstitutionName = 'Anonymized'
  #       # ds.PatientsBirthDate = ds.PatientsBirthDate[:4] + "0101"
  #       # print(subj_id, ds.PatientName, ds.PatientID, ds.InstitutionName)
  #       # output_path = dcm_path
  #       # if not os.path.exists(output_path):
  #       #   os.makedirs(output_path)
  #       pydicom.filewriter.write_file(dcm_path+dcm_file,ds) # write out anonymized file.
  # print('successfully anonymized',subj_id, 'anonymized patient name, id, and institution:',ds.PatientName, ds.PatientID, ds.InstitutionName)