#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:26:50 2019

@author: riccardo
"""
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch, torchvision
import os

DEFopt={'--outputtype':'atlas',
     '--overwrite':'False',
     '--N3':'False',
     '--multinet':'False',
     '--probmap':'False'}

labs=('Cortex','Hippocampus','Ventricles','Striatum','Background')

def GetFilesOptions(args,opt=DEFopt):
    VolumeList=[]
    optchange=''
    
    for k in args:
        if optchange!='':
            opt[optchange]=k
            optchange=''
            continue
            
        if '--' == k[0:2]:
            if k in opt:
                optchange=k
                continue
            else:
                raise NameError('Option '+k+' not recognized')
        
        VolumeList.append(k)
    
    
    if opt['--overwrite']=='False':
        opt['--overwrite']=False
    elif opt['--overwrite']=='True':
        opt['--overwrite']=True
    else:
        raise NameError('Value '+opt['--overwrite']+' unrecognized for option --overwrite')
    
    if opt['--probmap']=='False':
        opt['--probmap']=False
    elif opt['--probmap']=='True':
        opt['--probmap']=True
    else:
        raise NameError('Value '+opt['--probmap']+' unrecognized for option --probmap')
        
    if opt['--N3']=='False':
        opt['--N3']=False
    elif opt['--N3']=='True':
        opt['--N3']=True
    else:
        raise NameError('Value '+opt['--N3']+' unrecognized for option --N3')
    
    if opt['--outputtype'] not in ('atlas','4D','separate'):
        raise NameError('Value '+opt['--outputtype']+' unrecognized for option --outputtype')
        
    if opt['--multinet']=='False':
        opt['--multinet']=False
    elif opt['--multinet']=='True':
        opt['--multinet']=True
    else:
        raise NameError('Value '+opt['--multinet']+' unrecognized for option --multinet')
        
    return VolumeList, opt



def SaveNii(reference,volume,out_path,overwrite=False):
    if os.path.exists(out_path) and (not overwrite):
        raise NameError('File exists: '+out_path+'\nYou can also use --overwrite True')
    ref=nib.load(reference)
    Affine=ref.affine
    newnii=nib.Nifti1Image(volume,Affine)
    newnii.header['qoffset_x']=ref.header['qoffset_x']
    newnii.header['qoffset_y']=ref.header['qoffset_y']
    newnii.header['qoffset_z']=ref.header['qoffset_z']
    nib.save(newnii,out_path)

def SaveVolume(path,output,opt):
    Mask=output[0].detach().cpu().numpy()
    S=Mask.shape
    Mask=Mask.reshape((S[-3],S[-2],S[-1]))
    Labels=output[1].detach().cpu().numpy()
    out=os.path.splitext(path)[0]
    
    if not opt['--probmap']:
        Mask[Mask>=0.5]=1
        Mask[Mask<1]=0
        
        Labels[np.where(Labels== np.amax(Labels,axis=1))] = 1
        Labels[Labels!=1]=0
    

    SaveNii(path,Mask,out+'_Mask.nii.gz',opt['--overwrite'])
    for i in range(5):
        vol=np.zeros(Mask.shape)
        vol+=Labels[0,i,:,:,:]
        SaveNii(path,vol,out+'_'+labs[i]+'.nii.gz',opt['--overwrite'])
            
        

class Normalizer():
    
    def __call__(self,sample):
        mean=np.nanmean(sample['MRI'])
        std=np.nanstd(sample['MRI'])
        
        sample['MRI']=(sample['MRI']-mean)/(np.sqrt(std))
        
        return sample
    
class ToTensor():
    
    def __call__(self,sample):
        MRIT = torch.from_numpy(sample['MRI'])
        MRIT=MRIT.float().cuda()
        
        return {'MRI': MRIT}
    
normalizator = Normalizer()
tensorize = ToTensor()
transforms = torchvision.transforms.Compose([ normalizator, tensorize])
        
class SegmentUs(Dataset):
    def __init__(self, VolumeList,transform=transforms):
        """
        Turns the volume list in a torch dataset to infer through
        Just give it the volume list
        """
        self.dataset=[]
        self.paths=[]
        self.transform=transform
        for vol in VolumeList:
            if not os.path.isfile(vol):
                raise NameError (vol+' file does not exist')
            
            self.dataset.append(nib.load(vol))
            self.paths.append(  os.path.realpath(vol)  )
            
        
    def __getitem__(self,idx):
        S=self.dataset[idx].shape
        # MRI data
        MRI=self.dataset[idx].get_data()
        MRI=MRI.reshape([1, S[0], S[1], S[2]])
        
        
        sample = {'MRI': MRI}
        
        # Transform
        
        if self.transform:
            sample = self.transform(sample)
        
        sample['path']=self.paths[idx]
        return sample
        
    def __len__(self):
        return len(self.dataset)
    
