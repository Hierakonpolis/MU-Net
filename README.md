# MU-Net
Multi-task U-Net for the simultaneous segmentation and skull-stripping of mouse brain MRI

This convolutional neural network is designed to perform skull-stripping and region segmentation on mouse brain MRI. Files included:
MUNet.py: network definitions
runMU-Net.py : script to run segmentation on nifti MRI volumes
helperfile: Auxiliary functions for runMU-Net.py
weights: trained network parameters for each dataset fold
weights_N3: network parameters when trained on N3-corrected MRI volumes

Developed in PyTorch 1.0.1, the included script also requires nibabel, skimage and tqdm. 

This network is trained on coronal T2 mouse brain MRI delineated with a bounding box, and so for the network to function correctly MRI volumes need to be cropped to a bounding box around the brain. To automate this task we include a lightweight auxiliary network. You can exclude this step by using the "--boundingbox False" option.

# Usage:

python3 runMU-Net.py [options] [list of volumes]

[list of volumes] is a list of paths to nifti volumes separated by spaces

Options:

--overwrite [True/False]: Overwrite outputs if file already exists (default: False)
    
--N3 [True/False]: Load model weights for N3 corrected volumes (default False)
    
--multinet [True/False]: use networks trained on all folds and apply majority voting. (default True)

--probmap [True/False]: output unthresholded probability maps rather than the segmented volumes (default False)

--boundingbox [True/False]: automatically estimate bounding box using auxiliary network (default True)

--useGPU [True/False]: run on GPU, requires a CUDA enabled GPU and PyTorch installed with GPU support

Note: we assume the first two indices in the volume are contained in the same coronal section, so that the third index would refer to different coronal sections

