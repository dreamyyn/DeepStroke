# DeepStroke
This is a collection of basic code required for the deep stroke project.
The function is mostly explained by the file name. Here is a description of each file:
1. anonymize_dicom.py
  The code read through all dicom files under a path and anonymize those dicom files according to the requirement listed in the code.
2. createTMAXmask.py
  The code read through nii files of ADC/Tmax and create a new nii that only contain a masked ADC/Tmax by a specific threshold.
3.generate_dicom.py
  The code read output, ground truth, and FLAIR of the model, overlap the prediction and ground truth. and save image for each slice with comparison of Tmax/ADC segmentation.
