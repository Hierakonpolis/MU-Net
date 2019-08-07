#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:17:06 2019

@author: riccardo
"""
import sys

if len(sys.argv)==1:
    print('This script runs inference on MRI volumes for MU-Net:\nMulti-task U-Net for the simultaneous segmentation and skull-stripping of mouse brain MRI\n'+
          'This network is optimized to run on 2D coronal slices.\nHow to run:\n\n'+
          'python3 runMU-Net.py [options] [list of volumes]\n\n'+
          '[list of volumes] is a list of paths to nifti volumes separated by spaces\n'+
          'Options:'+
          '\n--overwrite [True/False]:\n'+
          '    Overwrite outputs if file already exists (default: False)\n'+
          '--N3 [True/False]:\n    Load model weights for N3 corrected volumes (default False)\n'+
          '--multinet [True/False]: use nets trained on all folds and apply majority voting. Must store all outputs in RAM (default False)\n'+
          '--probmap [True/False]: output unthresholded probability maps rather than the segmented volumes (default False)'+
          'Note: we assume the first two indices in the volume are contained in the same coronal section, so that the third index would refer to different coronal sections')
    exit()
    
import helperfile, torch, tqdm
import MUNet
    
VolumeList, opt = helperfile.GetFilesOptions(sys.argv[1:])

SegmentList=helperfile.SegmentUs(VolumeList)
SegmentLoader = torch.utils.data.DataLoader(SegmentList, batch_size=1, shuffle=False)
segmentations={}

if opt['--N3']:
    N3='_N3'
else:
    N3=''
    
for f in range(5):
    Net=MUNet.MUnet(MUNet.PARAMS_2D_NoSkip).cuda().eval()
    path='weights'+N3+'/Fold'+str(f+1)+N3+'/bestbyloss.tar'
    save=torch.load(path)
    Net.load_state_dict(save['model_state_dict'])
    if opt['--multinet']:
        print('Volume '+str(f+1)+'\n')
    
    for i, sample in tqdm.tqdm(enumerate(SegmentLoader)):
        with torch.no_grad():
            result=Net(sample['MRI'])
            spath=sample['path'][0]
            
            if opt['--multinet']:
                segmentations[f,i]=(spath,result)
            else:
                helperfile.SaveVolume(spath,result,opt)
    
    torch.cuda.empty_cache()
    del Net
    torch.cuda.empty_cache()
    
    if not opt['--multinet']: break

if opt['--multinet']:
    for i in range(len(SegmentLoader)):
        mask=segmentations[0,i][1][0]+segmentations[1,i][1][0]+segmentations[2,i][1][0]+segmentations[3,i][1][0]+segmentations[4,i][1][0]
        mask=mask/5
        labels=segmentations[0,i][1][1]+segmentations[1,i][1][1]+segmentations[2,i][1][1]+segmentations[3,i][1][1]+segmentations[4,i][1][1]
        labels=labels/5
        helperfile.SaveVolume(segmentations[0,i][0],(mask,labels),opt)