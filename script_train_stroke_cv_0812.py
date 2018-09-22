import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

''' basic dependencies '''
import numpy as np
import os
import json
import datetime
from time import time
import pydicom
import nibabel
import h5py

from subtle_fileio import *
from subtle_utils import *
from subtle_network import *
from subtle_metrics import *
from subtle_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

''' 
setup gpu
'''
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

'''
exclusion slices
'''
dict_slices_excluded = {}

''' 
get dataset info 
'''
dict_sample_info = {
                    '/data/yuanxie/stroke_preprocessing/':1.0,
                    }
ext_data = 'hdf5'
log_dir = '/data/yuanxie/log/'
list_volume_info = []

'''
pre setup parameters related to sampling
'''
keras_memory = 0.85
keras_backend = 'tf'
validation_split = 0.1
num_slice_expand = 2
axis_slice = 0


# related to computation perforamnce
num_workers = 16
max_queue_size = num_workers*2
# related to model
num_poolings = 3
num_conv_per_pooling = 3
# related to training
lr_init = 0.0005
batch_size = 16
# default settings
with_batch_norm = True
output_range = [0.,1.]
# dimension
num_slice_25d = num_slice_expand*2+1
index_slice_mid = num_slice_expand
num_contrast_input = 8
num_channel_input = num_slice_25d*num_contrast_input
num_channel_output = 1
shape_resize = [128,128]
img_rows = shape_resize[0]
img_cols = shape_resize[1]
# generator settings
generator_resize_data = False
generator_normalize_data = False
generator_mask_data = False
generator_sanitize_data = False
generator_augment_data = False
generator_axis_slice = 0
print('setup parameters')

''' 
setup model
'''
setKerasMemory(keras_memory)
loss_monitoring = [dice_coef, dice_coef_loss, seg_crossentropy, precision, recall]
# loss_func = dice_coef_loss
loss_func = seg_crossentropy

'''
model parameters
'''
model_parameters = {
                    "num_channel_input": num_channel_input,
                    "num_channel_output": num_channel_output,
                    "img_rows": img_rows,
                    "img_cols": img_cols,
                    "output_range": np.array(output_range),
                    "loss_function": loss_func,
                    "metrics_monitor": loss_monitoring,
                    "num_poolings": num_poolings, 
                    "num_conv_per_pooling": num_conv_per_pooling, 
                    "with_bn": with_batch_norm, 
                    "with_baseline_concat": True,
                    "with_baseline_addition": -1,
                    "activation_conv": 'relu',#'selu','relu','elu'
                    "activation_output": 'sigmoid', 
                    "kernel_initializer": 'he_normal',
                    "verbose": 1
                    }

model = deepResUNet(**model_parameters)

''' 
get files from each dataset 
'''
fold_cv = 5
for index_cv in range(0, fold_cv):
    print('########################### training fold #',index_cv+1)
    for dir_sample in dict_sample_info.keys():
      
        # get list of subject can be used
        list_subjects = sorted([os.path.join(dir_sample,x) for 
                                    x in os.listdir(dir_sample) if 
                                    os.path.isdir(os.path.join(dir_sample,x))])
        list_subjects_trainedon = []
        #print(list_subjects)
        num_subject = len(list_subjects)
        print('############################ total subject number:',num_subject)
        # loop through subjects
        for index_subject in range(num_subject):
            
            # get cases that should be used for training
            if index_subject%fold_cv == index_cv:
                continue
            dir_subject = list_subjects[index_subject]
            list_subjects_trainedon.append(dir_subject)
            # get file
            subject_name = dir_subject.split('/')[-1]
            #list_filename_inputs = sorted([x for x in os.listdir(dir_subject) if x.endswith(ext_data) and x.find('inputs_withALL')>=0 and x.find('without')<0])          
            list_filename_inputs = sorted([x for x in os.listdir(dir_subject) if x.endswith(ext_data) and x.find('inputs')>=0 and x.find('without')<0]) 
            list_filename_outputs = sorted([x for x in os.listdir(dir_subject) if x.endswith(ext_data) and x.find('output')>=0 and x.find('without')<0])                    
            len_inputs = len(list_filename_inputs)
            len_outputs = len(list_filename_outputs)

            # check valid file
            if len(list_filename_inputs)<=0 or len(list_filename_outputs)<=0:
                continue
            
            # loop pair of files
            for index_sample in [0, 2, 4, 6]:#range(len_inputs):
                filename_input = list_filename_inputs[index_sample%len_inputs]
                filename_output = list_filename_outputs[index_sample%len_outputs]        
                print('samples:', dir_subject, filename_input, filename_output)
                _ , data_shape, _ = get_data_from_ext(os.path.join(dir_subject, filename_input), ext_data, return_mean=False)
                values_mean = [-1,-1,-1]
                sample_weight = dict_sample_info[dir_sample]
                try:
                    slices_excluded = dict_slices_excluded[subject_name] 
                except:
                    slices_excluded = []
                list_volume_info.append({'filepath_inputs':[os.path.join(dir_subject, filename_input)],
                                             'filepath_output':os.path.join(dir_subject, filename_output), 
                                             'data_shape':data_shape, 
                                             'values_mean':values_mean,
                                             'sample_weight':sample_weight,
                                             'indexes_slice_excluded':slices_excluded})
        num_sample_file = len(list_volume_info)
        print('get {0} subject with {1} {2} file pairs'.format(
            num_subject, num_sample_file, ext_data))
        print('example:', list_volume_info[0])

    '''
    define samples
    '''
    list_samples_train = []
    for i in range(num_sample_file):
        # get sample info
        volume_info = list_volume_info[i]
        filepath_inputs = volume_info['filepath_inputs']
        filepath_output = volume_info['filepath_output']
        # dim
        data_shape = volume_info['data_shape']
        num_slices = data_shape[axis_slice]
        # value for normalization
        values_mean = volume_info['values_mean']
        sample_weight = volume_info['sample_weight']
        # get slices
        indexes_slice_excluded = volume_info['indexes_slice_excluded']
        indexes_slice_included = range(num_slice_expand, num_slices - num_slice_expand) 
        indexes_slice_included = sorted(list(set(indexes_slice_included) - set(indexes_slice_excluded)))
        # print(i, len(indexes_slice_excluded), len(indexes_slice_included), data_shape)
        # add to sample descriptions
        list_samples_train += [[filepath_inputs, 
                                filepath_output, 
                                x, sample_weight] for x in range(num_slice_expand, num_slices-num_slice_expand)]
    num_sample = len(list_samples_train)

    '''
    define train and validation
    '''
    # validation_split = 500
    np.random.seed(0)
    list_samples_train = [list_samples_train[x] for x in np.random.permutation(num_sample)]
    if validation_split>1:
        list_samples_validation = list_samples_train[-validation_split:]#.tolist()
        list_samples_train = list_samples_train[:int(num_sample-validation_split)]#.tolist()    
    else:
        list_samples_validation = list_samples_train[-int(num_sample*validation_split):]#.tolist()
        list_samples_train = list_samples_train[:int(num_sample*(1-validation_split))]#.tolist()
    print('train on {0} samples and validation on {1} samples'.format(
            len(list_samples_train), len(list_samples_validation)))


    '''
    setup model
    '''
    dir_ckpt = '/data/yuanxie/Enhao/ckpt_stroke'
    filename_init = 'model_stroke_sep18_cv{0}.ckpt'.format(index_cv)
    filename_checkpoint = 'model_stroke_sep18_cv{0}_run3.ckpt'.format(index_cv)
    filename_model = 'model_stroke_for_cv_relu_sigmoid_sep18.json'
    filepath_init = os.path.join(dir_ckpt, filename_init)
    filepath_checkpoint = os.path.join(dir_ckpt, filename_checkpoint)
    filepath_model = os.path.join(dir_ckpt, filename_model)
    print('setup parameters')


    '''
    init model
    '''
    # setup lr
    model.optimizer = Adam(lr = lr_init)
    print('train model:', filename_checkpoint)
    print('parameter count:', model.count_params())

    # setup weights
    try:
        model.load_weights(filepath_init)
        print('load model:', filepath_init)
    except:
        print('train from scratch')

    #setup callbacks
    model_checkpoint = ModelCheckpoint(filename_checkpoint, 
                                    monitor='val_loss', 
                                    save_best_only=True)        
    model_tensorboard = TensorBoard(log_dir="{0}/{1}_{2}".format(log_dir, 
                                time(), filename_checkpoint.split('.')[0]))
    model_callbacks = [model_checkpoint, model_tensorboard]


    '''
    save model to file
    '''
    model_json = model.to_json()
    with open(filepath_model, "w") as json_file:
        json_file.write(model_json)

    '''
    log for model setup
    '''
    print('train model description:', filepath_model)
    print('train model weights checkpoint:', filepath_checkpoint)
    print('parameter count:', model.count_params())

    '''
    define generator
    '''
    # details inside generator
    params_generator = {'dim_x': img_rows,
              'dim_y': img_cols,
              'dim_z': num_channel_input,
              'dim_25d' : num_slice_25d,
              'dim_output': 1,
              'batch_size': batch_size,
              'shuffle': True,
              'verbose': 1,
              'scale_data': 1,
              'scale_baseline': 0.0,
              'normalize_per_sample': False,
              'para': {                    
                       # 'index_mid':index_slice_mid,
                       # 'clip_output':[0,20],
                       # 'augmentation':[2,2,2,1],
                       'num_contrast_input':num_contrast_input
                       },          
              'normalize_data':generator_resize_data,
              'resize_data':generator_resize_data,        
              'mask_data':generator_mask_data,
              'sanitize_data':generator_sanitize_data,  
              'augment_data':generator_augment_data,  
              'axis_slice':generator_axis_slice
    }
    print('generator parameters:', params_generator)


    ''' 
    init generator
    '''
    training_generator = DataGenerator(**params_generator).generate(list_volume_info, list_samples_train)
    validation_generator = DataGenerator(**params_generator).generate(list_volume_info, list_samples_validation)        

    '''
    train model with generator
    '''
    num_epochs = 50
    steps_per_epoch = int(len(list_samples_train)/batch_size)
    validation_steps = int(len(list_samples_validation)/batch_size)
    print('train batch size:{0}, step per epoch:{1}, step per val epoch:{2}'.format(
        batch_size, steps_per_epoch, validation_steps))

    history = model.fit_generator(
                        generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=num_epochs,
                        callbacks = model_callbacks,
                        validation_data = validation_generator,
                        validation_steps = validation_steps,
                        max_queue_size=max_queue_size, 
                        workers=num_workers, 
                        use_multiprocessing=False
                        )   




