import h5py
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy import ndimage

def load_h5(path, key_h5='init', transpose_dims=[2, 0, 1, 3]):
    '''Loads h5 file and convert to a standard format.

    Parameters
    ----------
    path: Path to h5 file.

    Return
    ------
    numpy array.
    '''
    with h5py.File(path, 'r') as f:
        data = np.array(f[key_h5])

    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)
    if transpose_dims is not None:
        data = np.transpose(data, transpose_dims)

    return data
def crop(data,px_height=128,px_width=128):
    '''
    :param data: 3d img data
    :param px_height:
    :param px_width:
    :return:
    '''

    y_diff = (data.shape[2]-px_height)/2
    x_diff = (data.shape[1]-px_width)/2
    result = data[:,int(x_diff):int(x_diff)+px_width, int(y_diff):int(y_diff)+px_height,...]
    result = np.maximum(0, np.nan_to_num(result, 0))
    print(result.shape)
    return result

dir_stroke = '01002/'

path_ori = os.path.join(dir_stroke, 'inputs_aug0.hdf5')
data_ori = load_h5(path_ori, 'init', None)
# data_ori = np.transpose(data_ori,[1,2,0,3])
path_aug1 = os.path.join(dir_stroke, 'inputs_aug0.hdf5')
# data_aug1 = load_h5(path_aug1, 'init', None)
path_aug2 = os.path.join(dir_stroke, 'output_aug0.hdf5')
data_aug2 = load_h5(path_aug2, 'init', None)
# data_aug2 = np.transpose(data_aug2,[1,2,0,3])
path_aug3 = os.path.join(dir_stroke, 'output_aug0.hdf5')
# data_aug3 = load_h5(path_aug3, 'init', None)
# max = np.max(data_ori[:,:,:,0])
# print(max)
# pdb.set_trace()
# nonzero = np.nonzero(data_ori[:,:,:,2])
# data_new = data_ori[nonzero]
# print(np.mean(data_new))
# print(np.mean(data_ori[np.nonzero(data_ori[:,:,:,2])]))
# data_aug1 = ndimage.interpolation.rotate(data_ori,-30,axes=(1,2))
# print(data_aug1.shape,data_ori.shape)
# data_aug1 = crop(data_aug1,data_ori.shape[2],data_ori.shape[1])
# data_aug3 = ndimage.interpolation.rotate(data_aug2,-30,axes=(1,2))
# data_aug3 = crop(data_aug3,data_ori.shape[2],data_ori.shape[1])
# data_aug3 = (data_aug3 > 0.5) * 1.0
# print(np.max(data_aug3),np.mean(data_aug3))
# show output
f, ax = plt.subplots(1, 4, subplot_kw={'xticks': [], 'yticks': []})
ax[0].imshow(data_ori[35,:,:,0], cmap=plt.cm.gray)
ax[1].imshow(data_ori[35,:,:,1], cmap=plt.cm.gray)
ax[2].imshow(data_ori[35,:,:,2], cmap=plt.cm.gray)
ax[3].imshow(data_aug2[35,:,:,0], cmap=plt.cm.gray)
plt.show()
