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
          'Note: we assume the first two indices in the volume are contained in the same coronal section, so that the third index would refer to different coronal sections')
    exit()
import helperfile, torch, tqdm
import MUNet

#Extract options and MRI volumes list
VolumeList, opt = helperfile.GetFilesOptions(sys.argv[1:])
#Transforms for PyTorch tensors
normalizator = helperfile.Normalizer()
tensorize = helperfile.ToTensor(opt)
transforms = helperfile.torchvision.transforms.Compose([ normalizator, tensorize])
#Datasets for approximate mask inference
SegmentList=helperfile.SegmentUs(VolumeList,transform=transforms)
SegmentLoader = torch.utils.data.DataLoader(SegmentList, batch_size=1, shuffle=False)
segmentations={}
boxtemplates=[]
masks={}
pads=[]



print(sys.argv)
#This will modify paths based on using N3 bias corrected volumes or not
if opt['--N3']:
    N3='_N3'
else:
    N3=''
#Choose device
if opt['--useGPU']:
    
    stateloader = lambda savepath : torch.load(savepath)
else:
    stateloader = lambda savepath : torch.load(savepath,map_location='cpu')
    
# Build bounding boxes with auxiliary network, using all masky networks
if opt['--boundingbox']:
    print('Building bounding boxes\n')
    for f in range(5):
        with torch.no_grad():
            if opt['--useGPU']: 
                premask=MUNet.SkullNet(MUNet.PARAMS_SKULLNET).cuda().eval()
            else:
                premask=MUNet.SkullNet(MUNet.PARAMS_SKULLNET).cpu().eval()
            
            path='AuxW'+N3+'/Fold'+str(f+1)+N3+'/bestbyloss.tar'
            PMsave=stateloader(path)
            premask.load_state_dict(PMsave['model_state_dict'])
            for i, sample in tqdm.tqdm(enumerate(SegmentLoader),total=len(SegmentLoader)):
                
                masks[i,f]=premask(sample['MRI'])
            
            torch.cuda.empty_cache()
            del premask
            torch.cuda.empty_cache()
    
    for i in range(len(SegmentLoader)):
        tempo=masks[i,0]+masks[i,1]+masks[i,2]+masks[i,3]+masks[i,4]
        tempo=tempo.detach().cpu().numpy()/5
        tempo=tempo.reshape((tempo.shape[-3],tempo.shape[-2],tempo.shape[-1]))
        tempo[tempo>=0.5]=1
        tempo[tempo!=1]=0
        boxtemplates.append(helperfile.LargestComponent(helperfile.np.copy(tempo)))
else:
    boxtemplates=None
del masks

print('Segmenting...')
#final dataloader
SegmentList=helperfile.SegmentUs(VolumeList,boxtemplates=boxtemplates,transform=transforms)
SegmentLoader = torch.utils.data.DataLoader(SegmentList, batch_size=1, shuffle=False)

#Inference
for f in range(5):
    
    if opt['--useGPU']: 
        Net=MUNet.MUnet(MUNet.PARAMS_2D_NoSkip).cuda().eval()
    else:
        Net=MUNet.MUnet(MUNet.PARAMS_2D_NoSkip).cpu().eval()
        
    
    path='weights'+N3+'/Fold'+str(f+1)+N3+'/bestbyloss.tar'
    save=stateloader(path)
    Net.load_state_dict(save['model_state_dict'])
    if opt['--multinet']:
        print('Network '+str(f+1)+'\n')
    
    for i, sample in tqdm.tqdm(enumerate(SegmentLoader),total=len(SegmentLoader)):
        with torch.no_grad():
            result=Net(sample['MRI'])
            spath=sample['path'][0]
            pads.append(tuple(sample['offsets'][0].detach().cpu().numpy().astype(int)))
            if opt['--multinet']:
                segmentations[f,i]=(spath,result)
            else:
                helperfile.SaveVolume(spath,result,opt,pad=pads[i])
    
    torch.cuda.empty_cache()
    del Net
    torch.cuda.empty_cache()
    
    if not opt['--multinet']: break

if opt['--multinet']:
    for i in range(len(SegmentLoader)):
        mask=segmentations[0,i][1][0]+segmentations[1,i][1][0]+segmentations[2,i][1][0]+segmentations[3,i][1][0]+segmentations[4,i][1][0]
        mask/=5
        labels=segmentations[0,i][1][1]+segmentations[1,i][1][1]+segmentations[2,i][1][1]+segmentations[3,i][1][1]+segmentations[4,i][1][1]
        labels=labels/5
        
        helperfile.SaveVolume(segmentations[0,i][0],(mask,labels),opt,pad=pads[i])