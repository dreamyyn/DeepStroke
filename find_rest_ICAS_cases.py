import numpy as np
import glob
import shutil
import os

def glob_filename_list(path,keyword = '*',ext=''):
  filename_list = []
  for filename in glob.glob(path+keyword+ext):
    fn = filename.split('/')
    filename_list.append(fn[-1])
  return filename_list

filename_list = np.genfromtxt('ICAS_preocessed.txt', dtype=str)
path = '/Users/admin/deepstroke173/PWImasked185/'
local_list = glob_filename_list(path)
copypath = '/Users/admin/D2_RAPID/ICAS_RAPID/'
# for name in local_list:
for name in ['31001','31002','31003','31004','31005','31006','32001','32002','33002','33004','33005']:
  if int(name[:5])>30040 and (name[:5] not in filename_list):
    filepath = '/Users/admin/ICAS_origin/' + name[:5] + '/'
    timepoint = glob_filename_list(filepath)
    for i in range(len(timepoint)):
      scanpath = filepath + timepoint[i] +'/'
      dwiscan = glob_filename_list(scanpath, '*DWI*')
      diffscan = glob_filename_list(scanpath,'*DIFFUS*')
      dwi = dwiscan + diffscan
      # print(name, timepoint[i], dwiscan)
      if dwi:
        pwiscan = glob_filename_list(scanpath, '*PWI*')
        perfscan = glob_filename_list(scanpath, '*PERF*')
        pwi = pwiscan + perfscan
        if pwi:
          print(name, timepoint[i], dwi,pwi)
          if not os.path.exists(copypath + name[:5] +'/DWI/'):
            os.makedirs(copypath + name[:5] +'/DWI/')
          if not os.path.exists(copypath + name[:5] +'/PWI/'):
            os.makedirs(copypath + name[:5] +'/PWI/')
          for files in os.listdir(scanpath + dwi[0] +'/'):
            shutil.copy(scanpath + dwi[0] +'/' + files, copypath + name[:5] +'/DWI/')
          for files in os.listdir(scanpath + pwi[0] + '/'):
            shutil.copy(scanpath + pwi[0] + '/' + files, copypath + name[:5] + '/PWI/')
        else:
          print(name, 'does not have PWI in', timepoint[i])
          continue
        break
      else:
        print(name, 'does not have DWI in', timepoint[i])
        continue
