#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:17:06 2019

@author: riccardo
"""
import sys

if len(sys.argv)==1: # Helper message
    print('This script runs inference on MRI volumes for MU-Net:\nMulti-task U-Net for the simultaneous segmentation and skull-stripping of mouse brain MRI\n'+
          'This network is optimized to run on 2D coronal slices.\nHow to run:\n\n'+
          'python3 runMU-Net.py [options] [list of volumes]\n\n'+
          '[list of volumes] is a list of paths to nifti volumes separated by spaces\n'+
          'If a folder path is specified, all .nii and .ni.gz files will be added recursvely\n'+
          'Options:'+
          '\n--overwrite [True/False]:\n'+
          '    Overwrite outputs if file already exists (default: False)\n'+
          '--N3 [True/False]:\n    Load model weights for N3 corrected volumes (default False)\n'+
          '--multinet [True/False]: use networks trained on all folds and apply majority voting. (default True)\n'+
          '--probmap [True/False]: output unthresholded probability maps rather than the segmented volumes (default False)\n'+
          '--boundingbox [True/False]: automatically estimate bounding box using auxiliary network (default True)\n'+
          '--useGPU [True/False]: run on GPU, requires installed GPU support for pytorch with a CUDA enabled GPU (default True)\n'+
          '--namemask: only include files containing this string (case sensitive). Example: --namemask MySeq.nii\n'+
          '--nameignore: exclude all files containing this string (case sensitive). Example: --nameignore NotThisSeq.nii\n',
          '--out: output name, added to each output file\n',
          'Note: we assume the first two indices in the volume are contained in the same coronal section, so that the third index would refer to different coronal sections')
    exit()
import funcs

torch=funcs.torch

#Extract options and MRI volumes list
VolumeList, opt = funcs.GetFilesOptions(sys.argv[1:])




funcs.Segment(VolumeList,opt)