#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:40:22 2019

@author: riccardo
"""

import torch
import torch.nn as nn
import numpy as np

# Different sets of parameters for the network constructor


PARAMS_3D_NoSkip={'Categories':5, #Multiple categories, separated from 1 category task
                'FilterSize':int(5), # size of the convolution filters
                'FiltersNum':np.array([64, 64, 64, 64]), #number or tuple
                'ClassFilters':int(64), #number of filters in the classifier block
                'Depth':int(4),#number of dense blocks in each path # was 4
                'Activation':nn.LeakyReLU,  #Activation function
                'InblockSkip':False, #Use skip connections inside conv blocks
                'PoolShape':2, #Shape of the pooling operation
                # Convolutional operations modules
                'BNorm':nn.BatchNorm3d,
                'Conv':nn.Conv3d,
                'Pool':nn.MaxPool3d,
                'Unpool':nn.MaxUnpool3d
                }


PARAMS_3D_Skip={'Categories':5, 
                'FilterSize':int(5), 
                'FiltersNum':np.array([64, 64, 64, 64]), 
                'ClassFilters':int(64),
                'Depth':int(4),
                'Activation':nn.LeakyReLU, 
                'InblockSkip':True,
                'PoolShape':2,
                'BNorm':nn.BatchNorm3d,
                'Conv':nn.Conv3d,
                'Pool':nn.MaxPool3d,
                'Unpool':nn.MaxUnpool3d
                }

PARAMS_2D_Skip={'Categories':5, 
                'FilterSize':(5,5,1), 
                'FiltersNum':np.array([64, 64, 64, 64]),
                'ClassFilters':int(64), 
                'Depth':int(4),#n
                'Activation':nn.LeakyReLU, 
                'InblockSkip':True, 
                'PoolShape':(2,2,1),
                'BNorm':nn.BatchNorm3d,
                'Conv':nn.Conv3d,
                'Pool':nn.MaxPool3d,
                'Unpool':nn.MaxUnpool3d
                }

PARAMS_2D_NoSkip={'Categories':5,
                'FilterSize':(5,5,1),
                'FiltersNum':np.array([64, 64, 64, 64]),
                'ClassFilters':int(64),
                'Depth':int(4),
                'Activation':nn.LeakyReLU,
                'InblockSkip':False,
                'PoolShape':(2,2,1),
                'BNorm':nn.BatchNorm3d,
                'Conv':nn.Conv3d,
                'Pool':nn.MaxPool3d,
                'Unpool':nn.MaxUnpool3d
                }

PARAMS_3D_Skip_2DPOOL={'Categories':5,
                'FilterSize':int(5),
                'FiltersNum':np.array([64, 64, 64, 64]),
                'ClassFilters':int(64),
                'Depth':int(4),
                'Activation':nn.LeakyReLU,
                'InblockSkip':True,
                'PoolShape':(2,2,1),
                'BNorm':nn.BatchNorm3d,
                'Conv':nn.Conv3d,
                'Pool':nn.MaxPool3d,
                'Unpool':nn.MaxUnpool3d}

PARAMS_SKULLNET={'Categories':0,
                'FilterSize':(5,5,1),
                'FiltersNum':np.array([4, 8, 16, 32]),
                'ClassFilters':int(4),
                'Depth':int(4),
                'Activation':nn.LeakyReLU,
                'InblockSkip':False,
                'PoolShape':(2,2,1),
                'BNorm':nn.BatchNorm3d,
                'Conv':nn.Conv3d,
                'Pool':nn.MaxPool3d,
                'Unpool':nn.MaxUnpool3d}

PARAMS=PARAMS_2D_NoSkip
            
EPS=1e-10 # log offset to avoid log(0)

#Fix parameters
    
def FindPad(FilterSize):
    """
    Returns appropriate padding based on filter size
    """
    A=(np.array(FilterSize)-1)/2
    if type(FilterSize)==tuple:
        return tuple(A.astype(int))
    else:
        return int(A)
    
# One convolution step

class OneConv(nn.Module):
    """
    Performs one single convolution: activation of previous layer, batchnorm,
    convolution
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """


    def __init__(self,FilterIn,FilterNum, FilterSize,PAR):
        super(OneConv,self).__init__()
        # One activation - normalization - convolution step
        self.activate=PAR['Activation']()
        self.norm=PAR['BNorm'](int(FilterIn), eps=1e-05, momentum=0.1, affine=True)
        self.conv=PAR['Conv'](int(FilterIn),int(FilterNum),FilterSize,padding=FindPad(FilterSize) )
        
    def forward(self,layer):
        act=self.activate(layer)
        normalize=self.norm(act)
        convolve=self.conv(normalize)
        return convolve

# Bottleneck layer

class Bottleneck(nn.Module):
    """
    Bottleneck layer
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """


    def __init__(self,FilterIn,FilterNum,FilterSize,PAR):
        super(Bottleneck,self).__init__()
        self.norm=PAR['BNorm'](int(FilterIn), eps=1e-05, momentum=0.1, affine=True)
        self.conv=PAR['Conv'](int(FilterIn),int(FilterNum),FilterSize,padding=FindPad(FilterSize) )
        
    def forward(self,layer):
        normalize=self.norm(layer)
        convolve=self.conv(normalize)
        return convolve


# The type of convolution block will be chosen according to what is indicated
# in the parameters dictionary

class SkipConvBlock(nn.Module):
    """
    One full convolution block
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """

    def __init__(self,FilterIn,FilterNum,FilterSize,PAR):
        super(SkipConvBlock,self).__init__()
        self.conv1=OneConv(int(FilterIn),int(FilterNum),FilterSize=FilterSize,PAR=PAR)
        self.conv2=OneConv(int(FilterIn+FilterNum),int(FilterNum),FilterSize=FilterSize,PAR=PAR)
        self.conv3=OneConv(int(FilterIn+FilterNum*2),int(FilterNum),1,PAR=PAR)
        
    def forward(self,BlockInput):
        first=self.conv1(BlockInput)
        fconv=torch.cat((first,BlockInput),1)
        
        second=self.conv2(fconv)
        sconv=torch.cat((first,second,BlockInput),1)
        BlockOut=self.conv3(sconv)
        
        return BlockOut
    
class NoSkipConvBlock(nn.Module):
    """
    One full convolution block
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """

    def __init__(self,FilterIn,FilterNum,FilterSize,PAR):
        super(NoSkipConvBlock,self).__init__()
        self.conv1=OneConv(int(FilterIn),int(FilterNum),FilterSize=FilterSize,PAR=PAR)
        self.conv2=OneConv(int(FilterNum),int(FilterNum),FilterSize=FilterSize,PAR=PAR)
        self.conv3=OneConv(int(FilterNum),int(FilterNum),1,PAR=PAR)
        
    def forward(self,BlockInput):
        first=self.conv1(BlockInput)
        
        second=self.conv2(first)
        BlockOut=self.conv3(second)
        
        return BlockOut


def MonoLoss(Ytrue,Ypred):
    '''
    Returns binary cross entropy + dice loss for one 3D volume, normalized
    W0: scales added weight on region border
    W1: scales class weight for Binary Cross Entropy, should depend on
    class frequency
    '''
    DICE = -torch.div( torch.sum(torch.mul(torch.mul(Ytrue,Ypred),2)), torch.sum(torch.mul(Ypred,Ypred)) + torch.sum(torch.mul(Ytrue,Ytrue)) )
  
    return DICE

def CateLoss(Ytrue,Ypred):
    '''
    Categorical cross entropy + dice loss for multiple categories
    W0 is a scalar, scales weight on region border
    W1 is np.array([w1,w1,..]) of shape: (number of classes), should depend on
    class frequency
    '''
    
    DICE= -torch.div( torch.mul(2, torch.sum( torch.mul(Ytrue,Ypred)  )  ), torch.sum(torch.mul(Ypred,Ypred)) + torch.sum(torch.mul(Ytrue,Ytrue)) )
    
    return DICE

class Loss():
    """
    Overall loss function, you might need to tweak this based on how you build
    your data loader. 
    """
    def __init__(self,W0,W1,categoryW,categories=5):
        self.categories=categories

    def __call__(self,Ytrue,Ypred):
    
        Mask=Ytrue.narrow(1,0,1)
        PredMask=Ypred[0]
        Labels=Ytrue.narrow(1,1,self.categories)
        LabelsPred=Ypred[1]
        return MonoLoss(Mask,PredMask) + CateLoss(Labels,LabelsPred)


class SkullNet(nn.Module):
    
    def __init__(self,PARAMS=PARAMS_SKULLNET):
        super(SkullNet,self).__init__()
        self.PARAMS=PARAMS
        if PARAMS['InblockSkip']:
            ConvBlock=SkipConvBlock
            self.skipper=True
        else:
            ConvBlock=NoSkipConvBlock
            self.skipper=False
        self.layers=nn.ModuleDict()
        self.layers['Dense_Down'+str(0)]=ConvBlock(1,PARAMS['FiltersNum'][0],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        self.layers['Pool'+str(0)]=PARAMS['Pool'](PARAMS['PoolShape'],return_indices=True) 
        
        for i in range(1,PARAMS['Depth']):
            self.layers['Dense_Down'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i-1],PARAMS['FiltersNum'][i],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            self.layers['Pool'+str(i)]=PARAMS['Pool'](PARAMS['PoolShape'],return_indices=True) 
        
        self.layers['Bneck']=Bottleneck(PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        self.layers['Up'+str(i)]=PARAMS['Unpool'](PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][-1]+PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-2],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        for i in reversed(range(1,PARAMS['Depth']-1)):
            
            self.layers['Up'+str(i)]=PARAMS['Unpool'](PARAMS['PoolShape'])
            
            self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i]+PARAMS['FiltersNum'][i],PARAMS['FiltersNum'][i-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            
            
        self.layers['Up'+str(0)]=PARAMS['Unpool'](PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(0)]=ConvBlock(PARAMS['FiltersNum'][0]+PARAMS['FiltersNum'][0],PARAMS['ClassFilters'],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        self.layers['BinaryMask']=PARAMS['Conv'](PARAMS['ClassFilters'],1,1) #binary mask classifier
        self.sigmoid=nn.Sigmoid()
            
            
    def forward(self,MRI):
        pools={}
        dense={}
        dense[0] = self.layers['Dense_Down'+str(0)](MRI)
        dense[1], pools[0] = self.layers['Pool'+str(0)](dense[0])
        
        for i in range(1,self.PARAMS['Depth']):
            dense[i] = self.layers['Dense_Down'+str(i)](dense[i])
            dense[i+1], pools[i] = self.layers['Pool'+str(i)](dense[i])
        
        BotNeck = self.layers['Bneck'](dense[i+1])
        
        Updense={}
        Unpool={}
        
        Unpool[i] = self.layers['Up'+str(i)](BotNeck,pools[i],output_size=dense[i].size())
        cat=torch.cat([Unpool[i],dense[i]],dim=1)
        Updense[i] = self.layers['Dense_Up'+str(i)](cat)
        
        for i in reversed(range(self.PARAMS['Depth']-1)):
            
            Unpool[i]=self.layers['Up'+str(i)](Updense[i+1],pools[i],output_size=dense[i].size())
            cat=torch.cat([Unpool[i],dense[i]],dim=1)
            Updense[i]=self.layers['Dense_Up'+str(i)](cat)
        
        MonoClass=self.layers['BinaryMask'](Updense[0])
        
        Mask=self.sigmoid(MonoClass)
        
        return Mask 

class MUnet(nn.Module):
    """
    Network definition, without framing connections. 
    Returns (Mask,Classes)
    Generated based on parameters
    """
    
    def __init__(self,PARAMS=PARAMS):
        super(MUnet,self).__init__()
        self.PARAMS=PARAMS
        if PARAMS['InblockSkip']:
            ConvBlock=SkipConvBlock
            self.skipper=True
        else:
            ConvBlock=NoSkipConvBlock
            self.skipper=False
        self.layers=nn.ModuleDict()
        self.layers['Dense_Down'+str(0)]=ConvBlock(1,PARAMS['FiltersNum'][0],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        self.layers['Pool'+str(0)]=PARAMS['Pool'](PARAMS['PoolShape'],return_indices=True) 
        
        for i in range(1,PARAMS['Depth']):
            self.layers['Dense_Down'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i-1],PARAMS['FiltersNum'][i],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            self.layers['Pool'+str(i)]=PARAMS['Pool'](PARAMS['PoolShape'],return_indices=True) 
        
        self.layers['Bneck']=Bottleneck(PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        self.layers['Up'+str(i)]=PARAMS['Unpool'](PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][-1]+PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-2],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        for i in reversed(range(1,PARAMS['Depth']-1)):
            
            self.layers['Up'+str(i)]=PARAMS['Unpool'](PARAMS['PoolShape'])
            
            self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i]*2,PARAMS['FiltersNum'][i-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            
            
        self.layers['Up'+str(0)]=PARAMS['Unpool'](PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(0)]=ConvBlock(PARAMS['FiltersNum'][0]*2,PARAMS['ClassFilters'],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        

        self.layers['Classifier']=PARAMS['Conv'](PARAMS['ClassFilters'],PARAMS['Categories'],1) #classifier layer
        self.layers['BinaryMask']=PARAMS['Conv'](PARAMS['ClassFilters'],1,1) #binary mask classifier
        self.softmax=nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()
            
            
    def forward(self,MRI):
        pools={}
        dense={}
        dense[0] = self.layers['Dense_Down'+str(0)](MRI)
        dense[1], pools[0] = self.layers['Pool'+str(0)](dense[0])
        
        for i in range(1,self.PARAMS['Depth']):
            dense[i] = self.layers['Dense_Down'+str(i)](dense[i])
            dense[i+1], pools[i] = self.layers['Pool'+str(i)](dense[i])
        
        BotNeck = self.layers['Bneck'](dense[i+1])
        
        Updense={}
        Unpool={}
        
        Unpool[i] = self.layers['Up'+str(i)](BotNeck,pools[i],output_size=dense[i].size())
        cat=torch.cat([Unpool[i],dense[i]],dim=1)
        Updense[i] = self.layers['Dense_Up'+str(i)](cat)
        
        for i in reversed(range(self.PARAMS['Depth']-1)):
            
            Unpool[i]=self.layers['Up'+str(i)](Updense[i+1],pools[i],output_size=dense[i].size())
            cat=torch.cat([Unpool[i],dense[i]],dim=1)
            Updense[i]=self.layers['Dense_Up'+str(i)](cat)
            
        MultiClass=self.layers['Classifier'](Updense[0])
        MonoClass=self.layers['BinaryMask'](Updense[0])
        
        Mask=self.sigmoid(MonoClass)
        Classes=self.softmax(MultiClass)
        
        return Mask, Classes
    
class DualFrameMUnet(nn.Module):
    """
    Network definition, with framing connections
    Returns (Mask,Classes)
    Generated based on parameters
    """
    
    def __init__(self,PARAMS=PARAMS):
        super(DualFrameMUnet,self).__init__()
        if PARAMS['InblockSkip']:
            ConvBlock=SkipConvBlock
        else:
            ConvBlock=NoSkipConvBlock
        self.layers=nn.ModuleDict()
        self.layers['Dense_Down'+str(0)]=ConvBlock(1,PARAMS['FiltersNum'][0],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        self.layers['Pool'+str(0)]=PARAMS['Pool'](PARAMS['PoolShape'],return_indices=True) 
        
        for i in range(1,PARAMS['Depth']):
            self.layers['Dense_Down'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i-1],PARAMS['FiltersNum'][i],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            self.layers['Pool'+str(i)]=PARAMS['Pool'](PARAMS['PoolShape'],return_indices=True) 

        if PARAMS['Depth']==1: i=0
        self.layers['Bneck']=Bottleneck(PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        self.layers['Up'+str(i)]=PARAMS['Unpool'](PARAMS['PoolShape'])
        if PARAMS['Depth']!=1:
            self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][-1]*2,PARAMS['FiltersNum'][-2],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        for i in reversed(range(1,PARAMS['Depth']-1)):
            
            self.layers['Up'+str(i)]=PARAMS['Unpool'](PARAMS['PoolShape'])
            
            self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i]*2,PARAMS['FiltersNum'][i-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            
            
        self.layers['Up'+str(0)]=PARAMS['Unpool'](PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(0)]=ConvBlock(PARAMS['FiltersNum'][0]*2,PARAMS['ClassFilters'],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        

        self.layers['Classifier']=PARAMS['Conv'](PARAMS['ClassFilters'],PARAMS['Categories'],1) #classifier layer
        self.layers['BinaryMask']=PARAMS['Conv'](PARAMS['ClassFilters'],1,1) #binary mask classifier
        self.softmax=nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()
            
    def forward(self,MRI):
        pools={}
        pooled={}
        dense={}
        dense[0] = self.layers['Dense_Down'+str(0)](MRI)
        pooled[0], pools[0] = self.layers['Pool'+str(0)](dense[0])
        
        for i in range(1,PARAMS['Depth']):
            dense[i] = self.layers['Dense_Down'+str(i)](pooled[i-1])
            pooled[i], pools[i] = self.layers['Pool'+str(i)](dense[i])
        if PARAMS['Depth']==1: i=0
        BotNeck = self.layers['Bneck'](pooled[i])
        
        Updense={}
        Unpool={}
        
        Unpool[i] = self.layers['Up'+str(i)](BotNeck - pooled[i],pools[i],output_size=dense[i].size())
        
        cat=torch.cat([Unpool[i],dense[i]],dim=1)
        Updense[i] = self.layers['Dense_Up'+str(i)](cat)
        
        for i in reversed(range(PARAMS['Depth']-1)):
            
            Unpool[i]=self.layers['Up'+str(i)](Updense[i+1] - pooled[i],pools[i],output_size=dense[i].size())
            cat=torch.cat([Unpool[i],dense[i]],dim=1)
            Updense[i]=self.layers['Dense_Up'+str(i)](cat)
            
        MultiClass=self.layers['Classifier'](Updense[0])
        MonoClass=self.layers['BinaryMask'](Updense[0])
        
        Mask=self.sigmoid(MonoClass)
        Classes=self.softmax(MultiClass)
        
        return Mask, Classes
