#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:26:50 2019

@author: riccardo
"""
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch, torchvision, os, MUNet, tqdm, warnings
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label as SkLabel
#import matplotlib.pyplot as plt

DEFopt={'--overwrite':'False',
        '--N3':'False',
        '--multinet':'True',
        '--probmap':'False',
        '--boundingbox':'True',
        '--useGPU':'True',
        '--namemask':'',
        '--nameignore':'DEFAULT///IGNORE///STRING'}

labs=('Cortex','Hippocampus','Ventricles','Striatum','Background')

def Booler(opt,string):
    if type(opt[string]) == bool: return opt
    if opt[string].upper()=='FALSE':
        opt[string]=False
    elif opt[string].upper()=='TRUE':
        opt[string]=True
    else:
        raise NameError('Value '+opt[string]+' unrecognized for option '+string)
    return opt

def FillHoles(vol):
    S=vol.shape
    newvol=np.zeros_like(vol)
    for k in range(S[2]):
        newvol[:,:,k]=binary_fill_holes(vol[:,:,k])
    return newvol

def GetFilesOptions(args=[],opt=DEFopt):
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
        
        if os.path.isfile(k):
            VolumeList.append(k)
        elif os.path.isdir(k):
            for subdir, _, files in os.walk(k):
                for file in files:
                    if file.endswith('nii') or file.endswith('nii.gz'):
                        VolumeList.append(os.path.join(subdir,file))
        else:
            raise NameError('File or directory '+k+' not found')
    opt=Booler(opt,'--boundingbox')
    opt=Booler(opt,'--overwrite')
    opt=Booler(opt,'--probmap')
    opt=Booler(opt,'--N3')
    opt=Booler(opt,'--multinet')
    opt=Booler(opt,'--useGPU')
        
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
    Mask=output[0]#.detach().cpu().numpy()
    S=Mask.shape
    Mask=Mask.reshape((S[-3],S[-2],S[-1]))
    Labels=output[1]#.detach().cpu().numpy()
    out=os.path.splitext(path)[0]
    
    if not opt['--probmap']:
        Mask[Mask>=0.5]=1
        Mask[Mask<1]=0
        Mask=LargestComponent(Mask)
        Mask=FillHoles(Mask)
        
        Labels[np.where(Labels == np.amax(Labels,axis=1))] = 1
        Labels[Labels!=1]=0
        #Labels=Labels*Mask
    UNPM=Mask
    Mask=np.pad(Mask,pad)
    SaveNii(path,Mask,out+'_Mask.nii.gz',opt['--overwrite'])
    for i in range(5):
        if i==4:
            mul=1
        else:
            mul=UNPM
        vol=np.zeros(Mask.shape)
        vol+=np.pad(FillHoles(Labels[0,i,:,:,:]*mul),pad,'constant', constant_values=(padwith[i]))
        if (not opt['--probmap']) and (labs[i]!=labs[-1]): vol=vol*Mask
        SaveNii(path,vol,out+'_'+labs[i]+'.nii.gz',opt['--overwrite'])
            
def PaddingIsValid(pad,path):
    if np.any(np.array(pad)<0):
        warnings.warn('Something went wrong with the boxing of this volume, skipping '+path)
        return False
    else:
        return True

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
    def __init__(self, VolumeList,opt,boxtemplates=None,transform=None):
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
            if opt['--namemask'] in vol and opt['--nameignore'] not in vol:
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
    
    def remove(self,path):
        for x in range(len(self.paths)):
            if self.paths[x] == path: break
        del self.dataset[x]
        del self.paths[x]
    
def Segment(VolumeList,opt=None):
    if opt==None: opt=GetFilesOptions()[1]
    #Transforms for PyTorch tensors
    normalizator = Normalizer()
    tensorize = ToTensor(opt)
    transforms = torchvision.transforms.Compose([ normalizator, tensorize])
    #Datasets for approximate mask inference
    SegmentList=SegmentUs(VolumeList,opt,transform=transforms)
    SegmentLoader = torch.utils.data.DataLoader(SegmentList, batch_size=1, shuffle=False)
    segmentations={}
    boxtemplates=[]
    masks={}
    pads=[]
    toremove=[]
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
                    
                    masks[i,f]=premask(sample['MRI']).detach().cpu().numpy()
                
                torch.cuda.empty_cache()
                del premask
                torch.cuda.empty_cache()
        
        for i in range(len(SegmentList)):
            tempo=masks[i,0]+masks[i,1]+masks[i,2]+masks[i,3]+masks[i,4]
            tempo=tempo/5
            tempo=tempo.reshape((tempo.shape[-3],tempo.shape[-2],tempo.shape[-1]))
            tempo[tempo>=0.5]=1
            tempo[tempo!=1]=0
#            plt.imshow(tempo[:,:,10])
#            plt.show()
            if np.sum(tempo)==0: # look for invalid masks
                warnings.warn('Sample ignored: could not find brain volume for sample '+SegmentList[i]['path'],Warning)
                toremove.append(SegmentList[i]['path'])
            else:
                boxtemplates.append(LargestComponent(np.copy(tempo)))
    else:
        boxtemplates=None
    del masks
    
    # remove invalid subjects
    for x in toremove:
        SegmentList.remove(x)
    
    VolumeList=SegmentList.paths
    
    print('Segmenting...')
    #final dataloader
    SegmentList=SegmentUs(VolumeList,opt,boxtemplates=boxtemplates,transform=transforms)
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
                paddy=tuple(sample['offsets'][0].detach().cpu().numpy().astype(int))
                spath=sample['path'][0]
#                if not PaddingIsValid(paddy,spath): 
#                    segmentations[f,i]=(spath,(torch.zeros(1),torch.zeros(1)))
#                    pads.append((0,0))
#                else:
                result=list(Net(sample['MRI']))
                result[0]=result[0].detach().cpu().numpy()
                result[1]=result[1].detach().cpu().numpy()
                pads.append(paddy)
                if opt['--multinet']:
                    segmentations[f,i]=(spath,result)
                else:
                     
                    SaveVolume(spath,result,opt,pad=pads[i]) 
        
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
            SaveVolume(segmentations[0,i][0],(mask,labels),opt,pad=pads[i])