# from scipy import io as sio
# from scipy.ndimage import rotate
from scipy.misc import imsave
import numpy as np

import os
# import pydicom
import nibabel as nib

from skimage.transform import resize
from nibabel import processing
from skimage import data, color, io, img_as_float, morphology



'''
# load results based on .npz files
# standard format of our result.npz files are:
# gt - gold standard;
# output - predicted probability map;
# output_thres - 0.5 thresholded prediction map.
'''


# wait = input("PRESS ENTER TO CONTINUE.")

def create_mosaic(start_slice, end_slice, list_img, list_img_masked, row_num=5, spacing=2):
    '''
    takes in the starting slice #, end slice #, a list of original image and a list of masked image
    outputs the mosaic layout of original image, masked image in a fashion that 5 images are concatonated in a row
    in the img_both output, original images are put side-by-side to the masked images for comparison
    row_num - how many images we want in a row
    spacing - 1, append every image; 2, append every other image; so on so forth
    TODO: add automatic padding of zeros if the rest doesn't fill in a full 5 image row
    '''
    full_mosaic = np.array([])
    masked_mosaic = np.array([])
    for n in range(start_slice, end_slice, row_num * spacing):
        full_row = np.array([])
        masked_row = np.array([])
        for k in range(0, row_num * spacing, spacing):

            if (full_row.size == 0):
                # print('slice#: ', n+k, "------ a new row begins ---------")
                full_row = np.concatenate((list_img[n + k], list_img_masked[n + k]), axis=1)
                masked_row = list_img_masked[n + k]
            # print('fullrow_size: ', full_row.shape)
            # print('maskedrow_size: ', masked_row.shape)
            else:
                # print('slice#: ', n+k)
                # print('fullrow_size: ', full_row.shape)
                # print('maskedrow_size: ', masked_row.shape)
                full_row = np.concatenate((full_row, list_img[n + k], list_img_masked[n + k]), axis=1)
                masked_row = np.concatenate((masked_row, list_img_masked[n + k]), axis=1)

        # now append each row
        if (full_mosaic.size == 0):

            full_mosaic = full_row
            masked_mosaic = masked_row
        # print("------ first row of image completes ---------")
        # print('fullmosaic_size: ', full_mosaic.shape)
        # print('maskedmosaic_size: ', masked_mosaic.shape)
        else:

            full_mosaic = np.concatenate((full_mosaic, full_row), axis=0)
            masked_mosaic = np.concatenate((masked_mosaic, masked_row), axis=0)
        # print("------ another row of image completes ---------")
        # print('fullmosaic_size: ', full_mosaic.shape)
        # print('maskedmosaic_size: ', masked_mosaic.shape)

    return [full_mosaic, masked_mosaic]


def construct_colormap(rows, cols, ground_truth, prediction):
    '''
    - rows and cols are the input grount truth and prediction's size
    - ground_truth and prediction is assumed to be both 2D images
    - output color_mask is a 3D rgb numpy array
    '''
    color_mask = np.zeros((rows, cols, 3))
    for r in range(0, rows):
        for c in range(0, cols):
            # true positive, Green
            if (ground_truth[r, c] == 1 and prediction[r, c] == 1): color_mask[r, c, :] = [0, 1, 0]
            # false negative, Blue
            if (ground_truth[r, c] == 1 and prediction[r, c] == 0): color_mask[r, c, :] = [0, 0, 1]
            # false positive, Red + a little green = orange?????
            if (ground_truth[r, c] == 0 and prediction[r, c] == 1): color_mask[r, c, :] = [1, 0.2, 0]

    return color_mask


def formgrid(subj_id, data_dir, flair_path, name_tag='',low_lim = 10, up_lim = 60):
    proxy = nib.load(flair_path)
    flair_pad = proxy.get_fdata()
    rows, cols, slices = flair_pad.shape
    print('input image dimensions:')
    print('rows:', rows)
    print('cols:', cols)
    print('slices:', slices)
    list_img = []
    for s in range(0, slices):

        img_flair = np.squeeze(flair_pad[:, :, s])
        img_flair_n = (img_flair - np.min(img_flair)) / np.ptp(img_flair)
        img_flair_n = resize(img_flair_n, (rows, cols))
        # Construct RGB version of grey-level image
        img_flair_color = np.dstack((img_flair_n, img_flair_n, img_flair_n))
        img_flair_color = np.rot90(img_flair_color, 1)

        # store them in lists
        list_img.append(img_flair_color)

    [full_mosaic,masked_mosaic] = create_mosaic(low_lim, up_lim, list_img,list_img)

    # # show output
    # f, ax = plt.subplots(1, 2, subplot_kw={'xticks': [], 'yticks': []})
    # ax[0].imshow(full_mosaic, cmap=plt.cm.gray)
    # ax[1].imshow(masked_mosaic)
    # plt.show()

    # show only 1
    # plt.imshow(masked_mosaic)
    # plt.show()

    # # save as png
    # imsave(data_dir + subj_id + '_flair_masked.png',full_mosaic)
    imsave(data_dir + subj_id + name_tag + '.png', masked_mosaic)
def masked_grid(subj_id, data_dir, tmax_path, adc_path, T1mask_path, gt_path, flair_path, name_tag='', cluster = 125,low_lim = 10, up_lim = 60,tmax_mode=True):
    proxy = nib.load(flair_path)
    flair_pad = proxy.get_fdata()

    result = nib.load(adc_path)
    adc_smooth = processing.smooth_image(result, 3)
    result_volume = adc_smooth.get_fdata()

    tmax_data = nib.load(tmax_path)
    tmax_smooth = processing.smooth_image(tmax_data, 3)
    tmax = tmax_smooth.get_fdata()

    T1t = nib.load(T1mask_path)
    T1mask = T1t.get_fdata()

    gt = nib.load(gt_path)
    gt_volume = gt.get_fdata()

    rows, cols, slices = gt_volume.shape
    print('input image dimensions:')
    print('rows:', rows)
    print('cols:', cols)
    print('slices:', slices)

    result_volume = resize(result_volume, [rows, cols, slices])
    # gt_volume_trans = resize(gt_volume_trans,[rows,cols,slices])

    list_img_masked = []
    list_img = []

    for s in range(0, slices):
        # extract single slice of img from gold standard mask (gt)
        # predicted mask (thres) and raw image (flair)
        img_gt = 1.0 * (gt_volume[:, :, s] > thres_gt)
        # img_output = np.squeeze(vol_output_trans[s,:,:,0])
        T1mask_s = T1mask[:, :, s] > 0
        adc_result = (np.logical_and(result_volume[:, :, s] > 0, result_volume[:, :, s] <= thres_adc)) * T1mask_s
        if tmax_mode:
            tmax_result = (tmax[:, :, s] > thres_tmax) * T1mask_s
            img_output_thres = np.logical_or(adc_result, tmax_result)
        else:
            img_output_thres = adc_result
        img_output_thres = (morphology.remove_small_objects(img_output_thres, cluster)) * 1.0
        img_flair = np.squeeze(flair_pad[:, :, s])

        # Construct a color mask to superimpose
        color_mask = construct_colormap(rows, cols, img_gt, img_output_thres)

        # normalize img_flair to btw -1 and +1
        # img_flair_n = 2*(img_flair - np.min(img_flair))/np.ptp(img_flair)-1
        # normalize img_flair to btw 0 and +1
        img_flair_n = (img_flair - np.min(img_flair)) / np.ptp(img_flair)
        img_flair_n = resize(img_flair_n, (rows, cols))
        # Construct RGB version of grey-level image
        img_flair_color = np.dstack((img_flair_n, img_flair_n, img_flair_n))

        # Convert flair and color mask to Hue Saturation Value (HSV)
        # colorspace
        img_flair_hsv = color.rgb2hsv(img_flair_color)
        color_mask_hsv = color.rgb2hsv(color_mask)

        # Replace the hue and saturation of the original image
        # with that of the color mask - not sure what it means
        img_flair_hsv[..., 0] = color_mask_hsv[..., 0]
        img_flair_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

        img_flair_masked = color.hsv2rgb(img_flair_hsv)

        # pdb.set_trace()
        ## rotating images, remove or edit based on your situation

        img_flair_color = np.rot90(img_flair_color, 1)
        color_mask = np.rot90(color_mask, 1)
        img_flair_masked = np.rot90(img_flair_masked, 1)
        # store them in lists
        list_img_masked.append(img_flair_masked)
        list_img.append(img_flair_color)

    [full_mosaic, masked_mosaic] = create_mosaic(low_lim, up_lim, list_img, list_img_masked)

    # # show output
    # f, ax = plt.subplots(1, 2, subplot_kw={'xticks': [], 'yticks': []})
    # ax[0].imshow(full_mosaic, cmap=plt.cm.gray)
    # ax[1].imshow(masked_mosaic)
    # plt.show()

    # show only 1
    # plt.imshow(masked_mosaic)
    # plt.show()

    # # save as png
    # imsave(data_dir + subj_id + '_flair_masked.png',full_mosaic)
    imsave(data_dir + subj_id + name_tag + '.png', masked_mosaic)
###################### here's where the main begins ###########################
#list for testing, grouped based on TICI >=2b or <=2a
list_nonreper_test1 = ["03032","01020","01002","05002","08009","03045","11003","01042","30058A","30082A","08010","30049A","30073A"]
list_nonreper_test2 = ["09005","05011","05010","03002","01021","10002","30008A","30054","01010","30030A","01047","30032","03028"]
list_nonreper_test3 = ["09007","05006","11004","01017","03040","01007","05008","30007","01036","09006","09002","30043","10001"]
list_nonreper_test4 = ["03025","30037","09003","05007","05012","05003","30027A","05005","11002","10006","10007","30084A","02003"]
list_nonreper_test5 = ["01027","01038","01041","02005","01040","08007","10004","03018","09004","01004","01045","10005","11001"]

list_reper_test1 = ["30012","01028","30042","30116","03027","30006A","03017","30068","03043","03016","01003","03046","30099","30117","30046A","30102","03042","30028","03047","01029","01006"]
list_reper_test2 = ["08005","01048","30063","30101","05001","10003","03008","30069A","08008","30035A","03007","30041","30040","30096","30126","08003","30108","30097","03033","30120","01001"]
# 30018A belongs to test 3, but with no GRE. so temperarily removed. if tested in any combination with no GRE, should include 30018A.
list_reper_test3 = ["30018A","03036","30122","30048A","30053","03039","30098","30055","03026","01015","03037","30127","30078A","30071A","03024","01043","03011","30057A","30024A","30115","30002A"]
list_reper_test4 = ["03013","03035","08001","03031","30056","02004","03041","01032","03019","30103","30023","30061","03009","03020","30047A","30026A","30109","05009","03048","10009","30090A"]
list_reper_test5 = ["30092","30022A","30034","30106","30080","03001","01049","12001","30113","01044","30075A","30059A","03003","30077A","30045A","30124","02006","30001A"]

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
list_nonreper_test = list_nonreper_test1+list_nonreper_test2+list_nonreper_test3+list_nonreper_test4+list_nonreper_test5
list_reper_test = list_reper_test1+list_reper_test2+list_reper_test3+list_reper_test4+list_reper_test5

dir_save = '/Users/admin/stroke_DL/mosaic/'
inputs_list = ['DWI','ADC','PWITMAX','PWICBF','PWICBV','PWIMTT','FLAIR']
T1mask_path = '/Users/admin/controls_stroke_DL/001/T1.nii'
thres_gt = 0.9
alpha = 0.5
for subj_id in list_nonreper_test+list_reper_test:
    gt_path = '/Users/admin/deepstroke173/deepstroke173/' + subj_id + '/LESION.nii'
    flair_path = '/Users/admin/deepstroke173/deepstroke173/' + subj_id + '/FLAIR.nii'
    adc_path = '/Users/admin/deepstroke173/deepstroke173/' + subj_id + '/ADC.nii'
    for inputchannel in inputs_list:
        input_path = '/Users/admin/deepstroke173/deepstroke173/' + subj_id + '/' + inputchannel + '.nii'
        data_dir = dir_save + subj_id + '/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # formgrid(subj_id,data_dir,input_path,name_tag = inputchannel,low_lim=23, up_lim=73)
        #below form masked grid of tmax prediction or adc prediction:
        if 'ADC' in inputchannel:
            if subj_id in ['08005', '08008', '08009']:
                thres_adc = 310
            else:
                thres_adc = 620
            masked_grid(subj_id, data_dir, input_path, input_path, T1mask_path, gt_path, flair_path, name_tag= '_ADCpred',
                            cluster=10, low_lim=23, up_lim=73, tmax_mode=False)
        elif 'TMAX' in inputchannel:
            if subj_id in ['30073A', '30082A', '30084A']:
                # print(subject_name)
                thres_tmax = 6
            else:
                thres_tmax = 60
            masked_grid(subj_id, data_dir, input_path, adc_path, T1mask_path, gt_path, flair_path,
                            name_tag='_TMAX+ADCpred',
                            cluster=125, low_lim=23, up_lim=73, tmax_mode=True)