
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt

data_dir = '/Users/yuyannan/Documents/Study/neurology/StanfordRSL/stroke/results/review_cases/01007/prediction_0308_6inputs+thresholded_all_2c3p_U_bycase_dice_vol_lr0005_fold3_01007.nii'
proxy = nib.load(data_dir)
img_array = np.asarray(proxy.dataobj)
img_array = np.minimum(1, img_array)
img_array = np.maximum(0, img_array)

rows,cols,sls = img_array.shape
print("rows", rows)
print("cols", cols)
print("slices", sls)
cmap = 'CMRmap'
#
# for s in range(sls):
# 	print("using slice #", s)
# 	plt.imshow(img_array[:,:,s], cmap='CMRmap', interpolation='nearest')
# 	plt.show()
fig,ax = plt.subplots()
im = ax.imshow(img_array[:, :, 40],cmap='rainbow', interpolation='nearest')
cbar=ax.figure.colorbar(im,ax=ax)
cbar.ax.set_ylabel('probability',rotation =-90,va='bottom')
# plt.imshow(img_array[:, :, 26], cmap='gist_ncar', interpolation='nearest')
plt.show()