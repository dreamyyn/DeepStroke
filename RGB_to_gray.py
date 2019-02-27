import nibabel as nib
import pydicom


path = '/Users/admin/icas/30089/raw_files/1256_07192017/842/'
file = 'baseline.nii'
dcm = 'I0009.dcm'

# baseline_load = nib.load(path + file)
# print(baseline_load.shape)
ds = pydicom.dcmread(path+dcm)
ds.pixel_array = ds.pixel_array[0]
ds.PixelData = ds.pixel_array.tobytes()
ds.save_as(path + 'new' + dcm)