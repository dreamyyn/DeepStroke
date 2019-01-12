import numpy as np
import os

data_list = np.genfromtxt('Database_List.csv', delimiter=',', dtype=str)
filename_list = np.genfromtxt('list_patient.txt', dtype=str)
# print(filename_list[0])
# print(data_list[400,1])
# print(len(data_list))
path_data = '/Users/admin/UCLAdata/'
for i in range(0, len(filename_list)):
    filename = filename_list[i]
    path_patient = os.path.join(path_data, filename)
    patientid = filename[1:5]
    index = np.where(data_list[:, 1] == patientid)
    data_list_patient = np.squeeze(data_list[index, :])
    subfolder = [f.path for f in os.scandir(path_patient) if f.is_dir()]
    for folder in subfolder:
        try:
            scanid = folder.split(' ')[2]
            index_id = np.where(data_list_patient[:, 2] == scanid)
            filename_replace = data_list_patient[index_id, 0]
            # print('corres line', filename, scanid, filename_replace)
            print(folder,os.path.join(path_patient,filename_replace[0][0]))
            os.rename(folder, os.path.join(path_patient,filename_replace[0][0]))
        except:
            print(filename, 'data already processed or error encountered')