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
from skimage.measure import label as SkLabel

DEFopt={'--outputtype':'atlas',
        '--overwrite':'False',
        '--N3':'False',
        '--multinet':'True',
        '--probmap':'False',
        '--boundingbox':'True',
        '--useGPU':'True'}

labs=('Cortex','Hippocampus','Ventricles','Striatum','Background')

def Booler(opt,string):
    if opt[string].upper()=='FALSE':
        opt[string]=False
    elif opt[string].upper()=='TRUE':
        opt[string]=True
    else:
        raise NameError('Value '+opt[string]+' unrecognized for option '+string)
    return opt

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
    opt=Booler(opt,'--boundingbox')
    opt=Booler(opt,'--overwrite')
    opt=Booler(opt,'--probmap')
    opt=Booler(opt,'--N3')
    opt=Booler(opt,'--multinet')
    opt=Booler(opt,'--useGPU')
    
    if opt['--outputtype'] not in ('atlas','4D','separate'):
        raise NameError('Value '+opt['--outputtype']+' unrecognized for option --outputtype')
        
    return VolumeList, opt

padwith={0:0,
         1:0,
         2:0,
         3:0,
         4:1}

def LargestComponent(Mask):
    Divs=SkLabel(Mask)
    counts=np.zeros(np.max(Divs))
    for i in range(len(counts)):
        counts[i]=np.sum(Mask[Divs==(i+1)])
    ind=np.argmax(counts)+1
    Mask[Divs!=ind]=0
    return Mask

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

def SaveVolume(path,output,opt,pad=(0,0)):
    Mask=output[0].detach().cpu().numpy()
    S=Mask.shape
    Mask=Mask.reshape((S[-3],S[-2],S[-1]))
    Labels=output[1].detach().cpu().numpy()
    out=os.path.splitext(path)[0]
    
    if not opt['--probmap']:
        Mask[Mask>=0.5]=1
        Mask[Mask<1]=0
        Mask=LargestComponent(Mask)
        
        Labels[np.where(Labels == np.amax(Labels,axis=1))] = 1
        Labels[Labels!=1]=0
        Labels=Labels*Mask
    
    Mask=np.pad(Mask,pad)
    SaveNii(path,Mask,out+'_Mask.nii.gz',opt['--overwrite'])
    for i in range(5):
        vol=np.zeros(Mask.shape)
        vol+=np.pad(Labels[0,i,:,:,:],pad,'constant', constant_values=(padwith[i]))
        if (not opt['--probmap']) and (labs[i]!=labs[-1]): vol=vol*Mask
        SaveNii(path,vol,out+'_'+labs[i]+'.nii.gz',opt['--overwrite'])
            
        

class Normalizer():
    
    def __call__(self,sample):
        mean=np.nanmean(sample['MRI'])
        std=np.nanstd(sample['MRI'])
        
        sample['MRI']=(sample['MRI']-mean)/(np.sqrt(std))
        
        return sample
    
class ToTensor():
    
    def __init__(self,opt={'--useGPU':True}):
        self.cuda=opt['--useGPU']
    
    def __call__(self,sample):
        MRIT = torch.from_numpy(sample['MRI'])
        if self.cuda:
            MRIT=MRIT.float().cuda()
        else:
            MRIT=MRIT.float().cpu()
        sample['MRI']=MRIT
        return sample

def XYBox(Mask,offset=1):
    S=Mask.shape
    LowX=0
    while np.sum(Mask[LowX,:,:])==0:
        LowX+=offset
    
    LowY=0
    while np.sum(Mask[:,LowY,:])==0:
        LowY+=offset
    
    
    HighX=S[0]-1
    while np.sum(Mask[HighX,:,:])==0:
        HighX-=offset
    
    HighY=S[1]-1
    while np.sum(Mask[:,HighY,:])==0:
        HighY-=offset
        
    return LowX, HighX, LowY, HighY
        
class SegmentUs(Dataset):
    def __init__(self, VolumeList,boxtemplates=None,transform=None):
        """
        Turns the volume list in a torch dataset to infer through
        Just give it the volume list
        """
        self.dataset=[]
        self.paths=[]
        self.transform=transform
        self.boxtemplates=boxtemplates
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
        # Crop
        if self.boxtemplates!=None:
            
            LowX, HighX, LowY, HighY = XYBox(self.boxtemplates[idx])
            HighX, HighY = HighX + 1, HighY + 1
            
            sample['MRI']= sample['MRI'][:,LowX:HighX,LowY:HighY,:]
            offsets=((int(LowX), int(S[0]-HighX)),(int(LowY), int(S[1]-HighY)),(0,0))  #(0,0, int(LowY), int(S[1]-HighY), int(LowX), int(S[0]-HighX)) # (0,0, int(S[1]-HighY), int(LowY), int(S[0]-HighX), int(LowX))
        # Transform
        
        if self.transform:
            sample = self.transform(sample)
        
        sample['path']=self.paths[idx]
        if self.boxtemplates!=None: 
            sample['offsets']=np.array(offsets)
        return sample
        
    def __len__(self):
        return len(self.dataset)
    
