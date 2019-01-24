import h5py
import numpy as np

'''
this program load the h5 file into numpy array.
The h5 file has 3 "dataset":
1)'pred' as predicted probability, 
2)'label' as the 0/1 label of that voxel, 
3)and 'vxid' as voxel id (not related to its location on image thought)

'''
# input the path and name of h5 file & read h5 file.
read_path = '/Users/admin/stroke_DL/results/0113_6inputs+thresholded_all_4c2p_U_reperloss_weighted_binary_ce/'
train = h5py.File(read_path + '01017.h5','r')

pred = train.get('pred')
pred = np.array(pred)  # to change h5 object to numpy array.
label = train.get('label')
label = np.array(label)
id = train.get('vxid')
id = np.array(id)

print(pred[4000:4040])
print(label[4000:4040])
print(id[4000:4040])