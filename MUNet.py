#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:40:22 2019

@author: riccardo
"""

import torch
import torch.nn as nn
import numpy as np


PARAMS_3D_NoSkip={    'Categories':5, #Multiple categories, separated from 1 category task
            'FilterSize':int(5), #was 5
            'FiltersNum':np.array([64, 64, 64, 64]), #number or tuple
            'ClassFilters':int(64), #number of filters in the classifier block
            'Depth':int(4),#number of dense blocks in each path # was 4
            'Activation':nn.LeakyReLU,  #Activation function
            'InblockSkip':False, #Use skip connections inside conv blocks
            'ZBoundaries':False, #Look for region boudaries in the Z direction as well when calculating the loss function
            'PoolShape':2 #Shape of the pooling operation
            }

PARAMS_3D_Skip={    'Categories':5, 
            'FilterSize':int(5), 
            'FiltersNum':np.array([64, 64, 64, 64]), 
            'ClassFilters':int(64),
            'Depth':int(4),
            'Activation':nn.LeakyReLU, 
            'InblockSkip':True,
            'ZBoundaries':False,
            'PoolShape':2
            }

PARAMS_2D_Skip={    'Categories':5, 
            'FilterSize':(5,5,1), 
            'FiltersNum':np.array([64, 64, 64, 64]),
            'ClassFilters':int(64), 
            'Depth':int(4),#n
            'Activation':nn.LeakyReLU, 
            'InblockSkip':True, 
            'ZBoundaries':False, 
            'PoolShape':(2,2,1) #
            }

PARAMS_2D_NoSkip={    'Categories':5,
            'FilterSize':(5,5,1),
            'FiltersNum':np.array([64, 64, 64, 64]),
            'ClassFilters':int(64),
            'Depth':int(4),
            'Activation':nn.LeakyReLU,
            'InblockSkip':False,
            'ZBoundaries':False,
            'PoolShape':(2,2,1)
            }

PARAMS_3D_Skip_2DPOOL={    'Categories':5,
            'FilterSize':int(5),
            'FiltersNum':np.array([64, 64, 64, 64]),
            'ClassFilters':int(64),
            'Depth':int(4),
            'Activation':nn.LeakyReLU,
            'InblockSkip':True,
            'ZBoundaries':False,
            'PoolShape':(2,2,1)}

PARAMS=PARAMS_2D_NoSkip
            
EPS=1e-10 # log offset to avoid log(0)

torch.set_default_tensor_type('torch.cuda.FloatTensor') # t
torch.backends.cudnn.benchmark = True
#Fix parameters

if np.sum(np.array(PARAMS['FiltersNum']).shape)==0:
    PARAMS['FiltersNum']=np.ones((PARAMS['Depth']))*PARAMS['FiltersNum']
    PARAMS['FiltersNum']=PARAMS['FiltersNum'].astype(int)
# Convolution blocks with batchnorm and skip connections
    
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


    def __init__(self,FilterIn,FilterNum, FilterSize=PARAMS['FilterSize'],activation=PARAMS['Activation']):
        super(OneConv,self).__init__()
        # One activation - normalization - convolution step
        self.activate=activation()
        self.norm=nn.BatchNorm3d(int(FilterIn), eps=1e-05, momentum=0.1, affine=True)
        self.conv=nn.Conv3d(int(FilterIn),int(FilterNum),FilterSize,padding=FindPad(FilterSize) )
        
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


    def __init__(self,FilterIn,FilterNum,FilterSize=PARAMS['FilterSize']):
        super(Bottleneck,self).__init__()
        self.norm=nn.BatchNorm3d(int(FilterIn), eps=1e-05, momentum=0.1, affine=True)
        self.conv=nn.Conv3d(int(FilterIn),int(FilterNum),FilterSize,padding=FindPad(FilterSize) )
        
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

    def __init__(self,FilterIn,FilterNum,FilterSize=PARAMS['FilterSize']):
        super(SkipConvBlock,self).__init__()
        self.conv1=OneConv(int(FilterIn),int(FilterNum),FilterSize=FilterSize)
        self.conv2=OneConv(int(FilterIn+FilterNum),int(FilterNum),FilterSize=FilterSize)
        self.conv3=OneConv(int(FilterIn+FilterNum*2),int(FilterNum),1)
        
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

    def __init__(self,FilterIn,FilterNum,FilterSize=PARAMS['FilterSize']):
        super(NoSkipConvBlock,self).__init__()
        self.conv1=OneConv(int(FilterIn),int(FilterNum),FilterSize=FilterSize)
        self.conv2=OneConv(int(FilterNum),int(FilterNum),FilterSize=FilterSize)
        self.conv3=OneConv(int(FilterNum),int(FilterNum),1)
        
    def forward(self,BlockInput):
        first=self.conv1(BlockInput)
        
        second=self.conv2(first)
        BlockOut=self.conv3(second)
        
        return BlockOut

# Sobel filters, defined for the loss function. If no filter on
# anterior-posterior direction, it's set to 0
if PARAMS['ZBoundaries']:
    Z1=torch.tensor([[[ 1,  1, 1],
                     [ 1,  2, 1],
                     [ 1,  1, 1]],
            
                    [[ 0,  0, 0],
                     [ 0,  0, 0],
                     [ 0,  0, 0]],
            
                    [[ -1,  -1, -1],
                     [ -1,  -2, -1],
                     [ -1,  -1, -1]]],
                     requires_grad=False)
else:
    Z1=torch.tensor([[[ 0,  0, 0],
                     [ 0,  0, 0],
                     [ 0,  0, 0]],
            
                    [[ 0,  0, 0],
                     [ 0,  0, 0],
                     [ 0,  0, 0]],
            
                    [[ 0,  0, 0],
                     [ 0,  0, 0],
                     [ 0,  0, 0]]],
                     requires_grad=False)
X1=torch.tensor([[[ 1,  0, -1],
                     [ 1,  0, -1],
                     [ 1,  0, -1]],
            
                    [[ 1,  0, -1],
                     [ 2,  0, -2],
                     [ 1,  0, -1]],
            
                    [[ 1,  0, -1],
                     [ 1,  0, -1],
                     [ 1,  0, -1]]],
                     requires_grad=False)
Y1=torch.tensor([[[ 1,  1, 1],
                     [ 0,  0, 0],
                     [ -1,  -1, -1]],
            
                    [[ 1,  2, 1],
                     [ 0,  0, 0],
                     [ -1,  -2, -1]],
            
                    [[ 1,  1, 1],
                     [ 0,  0, 0],
                     [ -1,  -1, -1]]],
                     requires_grad=False)
    
# Define sobel filter function. This is all in format:
# [batch,channels,X,Y,Z]
    
def Sobel(Convolveme):
    
    """
    Sobel filters a volume in 3D
    """
    
    X=X1.reshape(1,1,3,3,3).type(torch.cuda.FloatTensor).expand(Convolveme.shape[1], -1,-1,-1,-1)
    Y=Y1.reshape(1,1,3,3,3).type(torch.cuda.FloatTensor).expand(Convolveme.shape[1], -1,-1,-1,-1)
    
    Z=Z1.reshape(1,1,3,3,3).type(torch.cuda.FloatTensor).expand(Convolveme.shape[1], -1,-1,-1,-1)
    Xconv=nn.functional.conv3d(Convolveme,X,groups=Convolveme.shape[1])
    Yconv=nn.functional.conv3d(Convolveme,Y,groups=Convolveme.shape[1])
    Zconv=nn.functional.conv3d(Convolveme,Z,groups=Convolveme.shape[1])
    conv=torch.abs(torch.nn.functional.pad(Xconv+Yconv+Zconv,(1,1,1,1,1,1)))
    conv[conv>0]=1
    
        
    return conv

def MonoLoss(Ytrue,Ypred,W0,W1):
    '''
    Returns binary cross entropy + dice loss for one 3D volume, normalized
    W0: scales added weight on region border
    W1: scales class weight for Binary Cross Entropy, should depend on
    class frequency
    '''
    shape=Ytrue.shape
    BCE = torch.mul((torch.ones(shape,requires_grad=False)-Ytrue), torch.log(torch.ones(shape,requires_grad=False)-Ypred + torch.ones(shape,requires_grad=False)*EPS))-torch.mul(Ytrue,torch.log(Ypred + torch.ones(shape,requires_grad=False)*EPS))
    W1b=torch.tensor(W1).reshape((1,1,1,1,1)).expand(shape).requires_grad=False
    W0b=torch.tensor(W0).reshape((1,1,1,1,1)).expand(shape).requires_grad=False
    W=W0b*Sobel(Ytrue)+W1b*torch.ones(shape,requires_grad=False)
    wBCE=torch.mul(W,BCE)
    mBCE = torch.mean(wBCE)
    DICE = -torch.div( torch.sum(torch.mul(torch.mul(Ytrue,Ypred),2)), torch.sum(torch.mul(Ypred,Ypred)) + torch.sum(torch.mul(Ytrue,Ytrue)) )
    loss = mBCE + DICE
    return loss

def CateLoss(Ytrue,Ypred,W0,W1,categories=PARAMS['Categories']):
    '''
    Categorical cross entropy + dice loss for multiple categories
    W0 is a scalar, scales weight on region border
    W1 is np.array([w1,w1,..]) of shape: (number of classes), should depend on
    class frequency
    '''
    shape=Ytrue.shape
    CCE=-torch.mul(Ytrue,torch.log(Ypred + torch.ones(shape,requires_grad=False)*EPS))
    W1b=torch.tensor(W1).reshape((1,categories,1,1,1)).expand(shape).requires_grad=False
    W0b=torch.tensor(W0).reshape((1,categories,1,1,1)).expand(shape).requires_grad=False
    W=W0b*Sobel(Ytrue)+W1b
    wCCE=torch.mul(W,CCE)
    DICE= -torch.div( torch.mul(2, torch.sum( torch.mul(Ytrue,Ypred)  )  ), torch.sum(torch.mul(Ypred,Ypred)) + torch.sum(torch.mul(Ytrue,Ytrue)) )
    mCCE=torch.mean(wCCE)
    loss= mCCE + DICE
    
    return loss

class Loss():
    """
    Overall loss function, you might need to tweak this based on how you build
    your data loader. 
    """
    def __init__(self,W0,W1,categoryW,categories=PARAMS['Categories']):
        self.W0=W0
        self.W1=W1
        self.categories=categories
        self.W=categoryW
    def __call__(self,Ytrue,Ypred):
    
        Mask=Ytrue.narrow(1,0,1)
        PredMask=Ypred[0]
        Labels=Ytrue.narrow(1,1,self.categories)
        LabelsPred=Ypred[1]
        Wm=self.W[0]
        Wl=self.W[1:]
        
        return MonoLoss(Mask,PredMask,self.W0*Wm,self.W1*Wm) + CateLoss(Labels,LabelsPred,self.W0*Wl,self.W1*Wl,self.categories)


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
        self.layers['Dense_Down'+str(0)]=ConvBlock(1,PARAMS['FiltersNum'][0],FilterSize=PARAMS['FilterSize'])
        self.layers['Pool'+str(0)]=nn.MaxPool3d(PARAMS['PoolShape'],return_indices=True) 
        
        for i in range(1,PARAMS['Depth']):
            self.layers['Dense_Down'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i-1],PARAMS['FiltersNum'][i],FilterSize=PARAMS['FilterSize'])
            self.layers['Pool'+str(i)]=nn.MaxPool3d(PARAMS['PoolShape'],return_indices=True) 
        
        self.layers['Bneck']=Bottleneck(PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-1],FilterSize=PARAMS['FilterSize'])
        
        self.layers['Up'+str(i)]=nn.MaxUnpool3d(PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][-1]+PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-2],FilterSize=PARAMS['FilterSize'])
        
        for i in reversed(range(1,PARAMS['Depth']-1)):
            
            self.layers['Up'+str(i)]=nn.MaxUnpool3d(PARAMS['PoolShape'])
            
            self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i]*2,PARAMS['FiltersNum'][i-1],FilterSize=PARAMS['FilterSize'])
            
            
        self.layers['Up'+str(0)]=nn.MaxUnpool3d(PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(0)]=ConvBlock(PARAMS['FiltersNum'][0]*2,PARAMS['ClassFilters'],FilterSize=PARAMS['FilterSize'])
        

        self.layers['Classifier'] = nn.Conv3d(PARAMS['ClassFilters'],PARAMS['Categories'],1) #classifier layer
        self.layers['BinaryMask'] = nn.Conv3d(PARAMS['ClassFilters'],1,1) #binary mask classifier
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
        self.layers['Dense_Down'+str(0)]=ConvBlock(1,PARAMS['FiltersNum'][0],FilterSize=PARAMS['FilterSize'])
        self.layers['Pool'+str(0)]=nn.MaxPool3d(PARAMS['PoolShape'],return_indices=True) 
        
        for i in range(1,PARAMS['Depth']):
            self.layers['Dense_Down'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i-1],PARAMS['FiltersNum'][i],FilterSize=PARAMS['FilterSize'])
            self.layers['Pool'+str(i)]=nn.MaxPool3d(PARAMS['PoolShape'],return_indices=True) 

        if PARAMS['Depth']==1: i=0
        self.layers['Bneck']=Bottleneck(PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-1],FilterSize=PARAMS['FilterSize'])
        
        self.layers['Up'+str(i)]=nn.MaxUnpool3d(PARAMS['PoolShape'])
        if PARAMS['Depth']!=1:
            self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][-1]*2,PARAMS['FiltersNum'][-2],FilterSize=PARAMS['FilterSize'])
        
        for i in reversed(range(1,PARAMS['Depth']-1)):
            
            self.layers['Up'+str(i)]=nn.MaxUnpool3d(PARAMS['PoolShape'])
            
            self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i]*2,PARAMS['FiltersNum'][i-1],FilterSize=PARAMS['FilterSize'])
            
            
        self.layers['Up'+str(0)]=nn.MaxUnpool3d(PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(0)]=ConvBlock(PARAMS['FiltersNum'][0]*2,PARAMS['ClassFilters'],FilterSize=PARAMS['FilterSize'])
        

        self.layers['Classifier'] = nn.Conv3d(PARAMS['ClassFilters'],PARAMS['Categories'],1) #classifier layer
        self.layers['BinaryMask'] = nn.Conv3d(PARAMS['ClassFilters'],1,1) #binary mask classifier
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
