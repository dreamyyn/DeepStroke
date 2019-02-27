
from scipy.misc import imsave
import numpy as np
import h5py
import os
# import pydicom
import nibabel as nib
from nibabel import processing
# import sys
# from glob import glob
from skimage import data, color, io, img_as_float,morphology

from skimage.transform import resize
import matplotlib.pyplot as plt


'''
# load results based on .npz files
# standard format of our result.npz files are:
# gt - gold standard;
# output - predicted probability map;
# output_thres - 0.5 thresholded prediction map.
'''
#wait = input("PRESS ENTER TO CONTINUE.")




def construct_colormap(rows,cols,ground_truth,prediction):
	'''
	- rows and cols are the input grount truth and prediction's size
	- ground_truth and prediction is assumed to be both 2D images
	- output color_mask is a 3D rgb numpy array
	'''
	color_mask = np.zeros((rows, cols, 3))
	for r in range(0,rows):
		for c in range(0,cols):
			# true positive, Green
			if (ground_truth[r,c] == 1 and prediction[r,c]==1): color_mask[r,c,:] = [0, 1, 0]
			# false negative, Blue
			if (ground_truth[r,c] == 1 and prediction[r,c]==0): color_mask[r,c,:] = [0, 0, 1]
			# false positive, Red + a little green = orange?????
			if (ground_truth[r,c] == 0 and prediction[r,c]==1): color_mask[r,c,:] = [1, 0.2, 0]

	return color_mask


def form_grid(result_volume, gt_volume, flair_pad, tmax_volume, adc_volume, dwi_mask,thres = 0.5, thres_gt =0.9,tmax_mode=True,thres_adc = 620, thres_tmax = 60):

	# extract single slice of img from gold standard mask (gt)
	# predicted mask (thres) and raw image (flair)
	img_gt = 1.0 * (gt_volume > thres_gt)
	# img_output = np.squeeze(vol_output_trans[s,:,:,0])
	img_output_thres = 1.0 * (result_volume > thres) * dwi_mask
	img_flair = np.squeeze(flair_pad)

	adc_result = (np.logical_and(adc_volume > 0, adc_volume <= thres_adc)) * dwi_mask
	if tmax_mode:
		tmax_result = (tmax_volume > thres_tmax) * dwi_mask
		img_output_tmax = np.logical_or(adc_result, tmax_result)
		cluster = 125
	else:
		img_output_tmax = adc_result
		cluster = 10
	img_output_tmax = (morphology.remove_small_objects(img_output_tmax, cluster)) * 1.0

	rows, cols = img_gt.shape
	# Construct a color mask to superimpose
	color_mask = construct_colormap(rows, cols, img_gt, img_output_thres)
	color_mask_tmax = construct_colormap(rows, cols, img_gt, img_output_tmax)

	# Construct RGB version of grey-level image
	img_flair_color = np.dstack((img_flair, img_flair, img_flair))

	# Convert flair and color mask to Hue Saturation Value (HSV)
	# colorspace
	img_flair_hsv = color.rgb2hsv(img_flair_color)
	color_mask_hsv = color.rgb2hsv(color_mask)
	color_mask_tmax = color.rgb2hsv(color_mask_tmax)

	# Replace the hue and saturation of the original image
	# with that of the color mask - not sure what it means
	img_flair_hsv[..., 0] = color_mask_hsv[..., 0]
	img_flair_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
	img_flair_masked = color.hsv2rgb(img_flair_hsv)

	img_flair_hsv[..., 0] = color_mask_tmax[..., 0]
	img_flair_hsv[..., 1] = color_mask_tmax[..., 1] * alpha
	img_flair_masked_tmax = color.hsv2rgb(img_flair_hsv)

	# pdb.set_trace()
	## rotating images, remove or edit based on your situation

	img_flair_color = np.rot90(img_flair_color, 1)
	color_mask = np.rot90(color_mask, 1)
	img_flair_masked = np.rot90(img_flair_masked, 1)
	img_flair_masked_tmax = np.rot90(img_flair_masked_tmax, 1)
	img_output = np.concatenate((img_flair_color,img_flair_masked, img_flair_masked_tmax),axis = 1)
	return img_output

def input_grid(dwi_volume, adc_volume, tmax_volume, mtt_volume, cbf_volume, cbv_volume):

	# if flip:
	# 	if subj_id[0]!= '3':
	# 		flair_pad = flair_pad[::-1,...]
	#
	# T1t = nib.load(T1mask_path)
	# T1mask = T1t.get_fdata() > 0 * 1.0
	# tmax = nib.load(tmax_path)
	# tmax_volume = tmax.get_fdata() * T1mask
	# mtt = nib.load(mtt_path)
	# mtt_volume = mtt.get_fdata() * T1mask
	# cbf = nib.load(cbf_path)
	# cbf_volume = cbf.get_fdata() * T1mask
	# cbv = nib.load(cbv_path)
	# cbv_volume = cbv.get_fdata() * T1mask
	# adc = nib.load(adc_path)
	# adc_volume = adc.get_fdata() * T1mask
	# dwi = nib.load(dwi_path)
	# dwi_volume = dwi.get_fdata() * T1mask
	# dwi_volume = np.maximum(0, np.nan_to_num(dwi_volume, 0))

	contrast_list = {'dwi': dwi_volume,'adc':adc_volume, 'tmax':tmax_volume, 'mtt':mtt_volume, 'cbf':cbf_volume,'cbv':cbv_volume}
	key_augments = list(contrast_list.keys())
	# print(key_augments)
	grid = {'dwi':[],'adc':[], 'tmax':[], 'mtt':[], 'cbf':[],'cbv':[]}
	for index in key_augments:
		img = contrast_list[index]
		img = np.squeeze(img[:, :])
		img_color = np.dstack((img, img, img))
		# Construct RGB version of grey-level image
		# img_color = np.dstack((img, img, img))
		## rotating images, remove or edit based on your situation
		# img_color = np.rot90(img_color, 1)
		img_color = np.rot90(img_color, 1)
		grid[index] = img_color
	img_input_line1 = np.concatenate((grid['dwi'],grid['adc'],grid['tmax']),axis = 1)
	img_input_line2 = np.concatenate((grid['mtt'], grid['cbf'], grid['cbv']), axis=1)
	return img_input_line1,img_input_line2

list_lg_test1 = ["30027A","30054","05010","30073A","01041","09005"]
list_lg_test2 = ["09002","30058A","10001","01040","03002","10005"]
list_lg_test3 = ["05002","05012","30082A","03025","01045","30084A"]
list_lg_test4 = ["01038","30030A","30049A","03032","10006","09003"]
list_lg_test5 = ["10007","01047","03018","05003","08010","11001","02003"]
list_lg_test = list_lg_test1, list_lg_test2, list_lg_test3, list_lg_test4, list_lg_test5

#list for testing, grouped based on TICI >=2b or <=2a
list_nonreper_test1 = ["03032","01020","01002","05002","08009","03045","11003","01042","30058A","30082A","08010","30049A","30073A","30019"]
list_nonreper_test2 = ["09005","05011","05010","03002","01021","10002","30008A","30054","01010","30030A","01047","30032","03028","30087"]
list_nonreper_test3 = ["09007","05006","11004","01017","03040","01007","05008","30007","01036","09006","09002","30043","10001"]
list_nonreper_test4 = ["03025","30037","09003","05007","05012","05003","30027A","05005","11002","10006","10007","30084A","02003"]
list_nonreper_test5 = ["01027","01038","01041","02005","01040","08007","10004","03018","09004","01004","01045","10005","11001"]

list_reper_test1 = ["30012","01028","30042","30116","03027","30006A","03017","30068","03043","03016","01003","03046","30099","30117","30046A","30102","03042","30028","03047","01029","01006","30044"]
list_reper_test2 = ["08005","01048","30063","30101","05001","10003","03008","30069A","08008","30035A","03007","30041","30040","30096","30126","08003","30108","30097","03033","30120","01001","30039"]
# 30018A belongs to test 3, but with no GRE. so temperarily removed. if tested in any combination with no GRE, should include 30018A.
list_reper_test3 = ["30018A","03036","30122","30048A","30053","03039","30098","30055","03026","01015","03037","30127","30078A","30071A","03024","01043","03011","30057A","30024A","30115","30002A","30016"]
list_reper_test4 = ["03013","03035","08001","03031","30056","02004","03041","01032","03019","30103","30023","30061","03009","03020","30047A","30026A","30109","05009","03048","10009","30090A","30015"]
list_reper_test5 = ["30092","30022A","30034","30106","30080","03001","01049","12001","30113","01044","30075A","30059A","03003","30077A","30045A","30124","02006","30001A","30005","30011","30014","30051"]

list_all_test1 = ["03043","30034","30124","30037","30069A","30058A","01027","30023","12001","03045","10002","08007","30109","05011","10009","01047","10005","30097","11001","10001","08009","30095","30117","11002","30113","03046","09005","30082A","30007","03007","30032","03048","01017","03019","30022A","30005","30011"]
# 08004 removed from test 2 because no lesion
list_all_test2 = ["03026","01001","05006","05008","30102","01049","01015","30080","30061","30045A","30054","08010","30068","03036","11003","09007","01042","10006","30012","30053","03042","09004","30063","01003","01038","30110","08001","02005","30101","30006A","01006","30127","03039","05007","30014","30015"]
list_all_test3 = ["03011","30057A","01040","03008","30024A","01002","03032","30042","03028","30108","05010","30104","03001","30084A","03025","30075A","30106","05002","30027A","01010","03003","30103","30047A","01007","03047","05009","03035","30049A","03018","30115","30028","03041","30002A","03024","01045","30016","30019"]
list_all_test4 = ["03009","01041","03037","30090A","02004","01028","30041","01020","30025","02001","30040","30122","30077A","09002","30120","03013","02003","05003","30098","01021","08008","11004","30008A","05012","03033","01048","30099","30071A","01044","30030A","30048A","30055","30018A","03020","03031","30039","30044"]
list_all_test5 = ["03017","30035A","30001A","01043","30096","01004","10003","08005","30073A","30056","05001","03027","09006","30100","08003","10004","30116","30092","02006","30026A","30046A","03002","03040","09003","30059A","30043","01029","03016","30078A","10007","05005","01036","30126","01032","30051","30087"]
list_all_test = list_all_test1, list_all_test2, list_all_test3, list_all_test4, list_all_test5
# grouped based on reperfusion rate <30% or >70%
# list_nonreper_test1 = ["30032","01041","30006A","10006","05010","30054","08007"]
# list_nonreper_test2 = ["01020","03040","03008","01017","03001","01040","30058A"]
# list_nonreper_test3 = ["11001","10005","11002","30082A","05002","30027A","03041"]
# list_nonreper_test4 = ["30043","03027","30037","03032","01027","30073A","01021"]
# list_nonreper_test5 = ["11003","01003","10002","09004","05005","30049A","09002"]
# list_reper_test1 = ["30077A","30108","08010","03042","09005","30023","30059A","30045A","08005","03035","03020","30061","03036","10009"]
# list_reper_test2 = ["03048","08008","03033","01036","03037","30101","30022A","05011","30098","08003","30056","05012","30122","05001"]
# list_reper_test3 = ["30041","10004","30028","30047A","09007","02005","05009","30046A","30126","30042","09006","10003","30002A","01010"]
# list_reper_test4 = ["12001","30055","03043","03031","03013","30124","03019","01029","30035A","08001","03047","30127","03028","01015"]
# list_reper_test5 = ["30113","01006","30075A","30001A","03026","30026A","03011","10007","03024","01044","01043","03009","03016","30096"]
list_nonreper_test = list_nonreper_test1,list_nonreper_test2,list_nonreper_test3,list_nonreper_test4,list_nonreper_test5
list_reper_test = list_reper_test1,list_reper_test2,list_reper_test3,list_reper_test4,list_reper_test5
###################### here's where the main begins ###########################
alpha = 0.6 # color hue
thres = 0.5
thres_gt = 0.9
flip = False
# model_name ='1129_charlesmod_DWI+ADC_reper_aug_true0.9_nonan'
model_name = '0124_6inputs+thresholded_all_4c2p_U_98weighted_binary_ce'
name_tag = '_all' + str(thres)
data_dir = '/Users/admin/stroke_DL/results/' + model_name +'/' # On longo:'/data3/yuanxie/project_stroke_mask/'
lower_lim = 0
upper_lim = 60

# if 'nonreper' in name_tag:
# 	tmax_mode = True
# elif 'reper' in name_tag:
# 	tmax_mode = False
# print(tmax_mode)
for cv in range(0,5):
	for subj_id in list_all_test[cv]:
	# for subj_id in ['01020']:
		# subj_id = subj_id[:5]
		print(subj_id)
		dwi_path = '/Users/admin/deepstroke173/PWImasked185/' + subj_id + '/DWI.nii'
		tmax_path = '/Users/admin/deepstroke173/PWImasked185/' + subj_id + '/PWITMAX.nii'
		adc_path = '/Users/admin/deepstroke173/PWImasked185/' + subj_id + '/ADC.nii'
		mtt_path = '/Users/admin/deepstroke173/PWImasked185/' + subj_id + '/PWIMTT.nii'
		cbf_path = '/Users/admin/deepstroke173/PWImasked185/' + subj_id + '/PWICBF.nii'
		cbv_path = '/Users/admin/deepstroke173/PWImasked185/' + subj_id + '/PWICBV.nii'
		T1mask_path = '/Users/admin/controls_stroke_DL/001/T1_cerebrum.nii'
		result_path = data_dir + 'prediction_' + model_name + '_fold{0}_'.format(cv + 1) + subj_id + '.nii'
		gt_path = data_dir + 'gt_' + model_name + '_fold{0}_'.format(cv + 1) + subj_id + '.nii'
		flair_path = data_dir + 'flair_' + model_name + '_fold{0}_'.format(cv + 1) + subj_id + '.nii'

		if not os.path.exists(gt_path):
			print('no file for', subj_id)
			continue

		proxy = nib.load(flair_path)
		flair_pad = proxy.get_fdata()
		if flip:
			if subj_id[0] != '3':
				flair_pad = flair_pad[::-1, ...]
		result = nib.load(result_path)
		result_volume = result.get_fdata()

		gt = nib.load(gt_path)
		gt_volume = gt.get_fdata()

		T1t = nib.load(T1mask_path)
		T1mask = T1t.get_fdata() > 0 * 1.

		dwi = nib.load(dwi_path)
		dwi_volume = dwi.get_fdata()
		dwi_volume = np.maximum(0, np.nan_to_num(dwi_volume, 0))
		dwi_volume = dwi_volume * T1mask
		dwi_volume = dwi_volume[:,:,lower_lim:upper_lim]
		mean_dwi = np.mean(dwi_volume[np.nonzero(dwi_volume)])
		dwi_mask = dwi_volume > (0.3 * mean_dwi)
		adc = nib.load(adc_path)
		adc_volume = adc.get_fdata() * T1mask
		adc_volume = adc_volume[:,:,lower_lim:upper_lim]
		tmax_data = nib.load(tmax_path)
		tmax = processing.smooth_image(tmax_data, 3)
		tmax_volume = tmax.get_fdata() * T1mask
		tmax_volume = tmax_volume[:,:,lower_lim:upper_lim]
		mtt = nib.load(mtt_path)
		mtt_volume = mtt.get_fdata() * T1mask
		mtt_volume = mtt_volume[:,:,lower_lim:upper_lim]
		cbf = nib.load(cbf_path)
		cbf_volume = cbf.get_fdata() * T1mask
		cbf_volume = cbf_volume[:,:,lower_lim:upper_lim]
		cbv = nib.load(cbv_path)
		cbv_volume = cbv.get_fdata() * T1mask
		cbv_volume = cbv_volume[:,:,lower_lim:upper_lim]
		T1mask = T1mask[:,:,lower_lim:upper_lim]
		img_list = {'dwi': dwi_volume, 'adc': adc_volume, 'tmax': tmax_volume,
										 'mtt': mtt_volume, 'cbf': cbf_volume,
										 'cbv': cbv_volume, 'flair': flair_pad, 'model': result_volume,
										 'gt': gt_volume, 'mask': dwi_mask}
		img_list['flair'] = (img_list['flair'] - np.min(img_list['flair'])) / np.ptp(img_list['flair'])
		input_list = {'dwi': dwi_volume, 'adc': adc_volume, 'tmax': tmax_volume,
										 'mtt': mtt_volume, 'cbf': cbf_volume,
										 'cbv': cbv_volume}
		key_img = list(input_list.keys())
		for key in key_img:
			img = input_list[key]
			# print(img)
			img_n = (img - np.min(img)) / np.ptp(img)
			input_list[key] = img_n

		if int(subj_id[:5]) >= 30059:
			# print(subject_name)
			thres_tmax = 6
		else:
			thres_tmax = 60

		if subj_id in ['08005', '08008', '08009']:
			thres_adc = 310
		else:
			thres_adc = 620

		for s in range(gt_volume.shape[2]):
			contrast_list = {'dwi': img_list['dwi'][:,:,s], 'adc': img_list['adc'][:,:,s], 'tmax': img_list['tmax'][:,:,s], 'mtt': img_list['mtt'][:,:,s], 'cbf': img_list['cbf'][:,:,s],
											 'cbv': img_list['cbv'][:,:,s], 'flair': img_list['flair'][:,:,s], 'model': img_list['model'][:,:,s], 'gt': img_list['gt'][:,:,s], 'mask': img_list['mask'][:,:,s]}
			contrast_list_input = {'dwi': input_list['dwi'][:,:,s], 'adc': input_list['adc'][:,:,s], 'tmax': input_list['tmax'][:,:,s],
										 'mtt': input_list['mtt'][:,:,s], 'cbf': input_list['cbf'][:,:,s],'cbv': input_list['cbv'][:,:,s]}
			img_output_tmax = form_grid(contrast_list['model'], contrast_list['gt'], contrast_list['flair'], contrast_list['tmax'],
														 contrast_list['adc'], contrast_list['mask'], thres=thres, thres_gt=thres_gt,
														 tmax_mode=True, thres_adc=thres_adc, thres_tmax=thres_tmax)
			img_output_adc = form_grid(contrast_list['model'], contrast_list['gt'], contrast_list['flair'], contrast_list['tmax'],
														 contrast_list['adc'], contrast_list['mask'], thres=thres, thres_gt=thres_gt,
														 tmax_mode=False, thres_adc=thres_adc, thres_tmax=thres_tmax)
			# img_input_line1, img_input_line2 = input_grid(contrast_list_input['dwi'], contrast_list_input['adc'], contrast_list_input['tmax'], contrast_list_input['mtt'], contrast_list_input['cbf'], contrast_list_input['cbv'])
			# print(img_output.shape, img_input_line2.shape)
			# img_input = np.concatenate((img_input_line1, img_input_line2), axis=0)
			img_output = np.concatenate((img_output_tmax,img_output_adc[:,int(img_output_adc.shape[1]*2/3):]), axis =1)
			# img_final = np.concatenate((img_input_line1,img_input_line2,img_output),axis = 0)
			output_path = data_dir + 'review' + name_tag + '/' + subj_id + '/'
			if not os.path.exists(output_path):
				os.makedirs(output_path)
			imsave(output_path + subj_id + '_{0:03}.png'.format(s), img_output)
		print('successfully saved' , subj_id)
