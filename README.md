# MU-Net
Multi-task U-Net for the simultaneous segmentation and skull-stripping of mouse brain MRI

This convolutional neural network is designed to perform skull-stripping and region segmentation on mouse brain MRI. Files included:
MUNet.py: network definitions
runMU-Net.py : script to run segmentation on nifti MRI volumes
helperfile: Auxiliary functions for runMU-Net.py
weights: trained network parameters for each dataset fold
weights_N3: network parameters when trained on N3-corrected MRI volumes

Developed in PyTorch 1.0.1, the included script also requires nibabel and tqdm

This network is trained on coronal T2 mouse brain MRI. For the network to function correctly, please crop your MRI volumes by delineating a bounding box around the brain.
