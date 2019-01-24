import numpy as np
import glob

def glob_filename_list(path,ext=''):
  filename_list = []
  for filename in glob.glob(path+'*'+ext):
    fn = filename.split('/')
    filename_list.append(fn[-1])
  return filename_list

filename_list = np.genfromtxt('ICAS_preocessed.txt', dtype=str)
path = '/Users/admin//'
local_list = glob_filename_list(path)
for name in local_list:
  if name not in filename_list:
    filepath = path + name + '/'
    timepoint = glob_filename_list(filepath)
    print(name, timepoint)
