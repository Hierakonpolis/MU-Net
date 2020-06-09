#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:43:25 2019

@author: riccardo
"""

import funcs as H
# import Helper2 as H2
import numpy as np
import torch
import torchvision
from MUNet import MUnet, PARAMS
import time
import tensorboardX as tbX
import os
from torch.optim import Adam

def MonoLoss(Ytrue,Ypred):
    return -torch.div( torch.sum(torch.mul(torch.mul(Ytrue,Ypred),2)), torch.sum(torch.mul(Ypred,Ypred)) + torch.sum(torch.mul(Ytrue,Ytrue)) )
    

def CateLoss(Ytrue,Ypred):
    return -torch.div( torch.mul(2, torch.sum( torch.mul(Ytrue,Ypred)  )  ), torch.sum(torch.mul(Ypred,Ypred)) + torch.sum(torch.mul(Ytrue,Ytrue)) )
    

class Loss():
    def __init__(self,categories):
        self.categories=categories
    def __call__(self,Ytrue,Ypred):
    
        Mask=Ytrue.narrow(1,0,1)
        PredMask=Ypred[0]
        Labels=Ytrue.narrow(1,1,self.categories)
        LabelsPred=Ypred[1]
        
        return MonoLoss(Mask,PredMask) + CateLoss(Labels,LabelsPred)

folds=5
decay=0.00
maxtime=12 *60*60 # hours

Network=MUnet

PAR=PARAMS

name='RunName'
SemW=0
DiceW=1
BBalance=False
crop=False
CCEW=0
Lossf=Loss(PAR['Categories'])

folder='dataset/path'
saveprogress='path/save.tar'
   
CurrentFold=1


while CurrentFold <= folds:
    scaler = H.Rescale(1.01,0.95,0.5) 
    rotator = H.RotCoronal(5,0.5)
    normalizator = H.Normalizer()#'/mnt/90FACB14FACAF60E/U-Net/Data/CRL0616_01_DATA_ANALYSIS')
    tensorize = H.ToTensor()
    transforms = torchvision.transforms.Compose([rotator, scaler, normalizator, tensorize])
    transforms_eval = torchvision.transforms.Compose([normalizator, tensorize])
    training_set=H.RodentDatasetCR(folder,fold=CurrentFold,FoldType='Train',transform=transforms)
    test_set=H.RodentDatasetCR(folder,fold=CurrentFold,FoldType='Test',transform=transforms_eval)

    dataloader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    
    run_name='Fold'+str(CurrentFold)+name
    runfolder='runs/'+run_name
    savefile='saves/'+run_name+'/bestbyloss.tar'
    # 'Fold'+str(CurrentFold)+'_3D_Vanilla_Skip'
    
    
    #Lossf=Loss(BoundaryWeight,DiceWeight,W)
    step=0
    os.mkdir('saves/'+run_name)    
        
    UN=Network().cuda()
    TotalTime=0
    epoch=0
    BestStep=0
    bestlossstep=0
    bestloss=0
    bestdice=0
    optimizer=Adam(UN.parameters(),weight_decay=decay)
    savedict={'dices':[],
              'losses':[],
              'mean_dices':[],
              'mean_loss':[],
              'step':[]}
    
    
    UN.train()
    
    writer=tbX.SummaryWriter(runfolder)
    
    time_check=time.time()
    
    while TotalTime < maxtime and (step - BestStep < 3000):
        dice=[]
        print('Epoch '+str(epoch))
        
        for i, sample in enumerate(dataloader):
            step+=1
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            
            Labels = UN(sample['MRI'])
            loss = Lossf(sample['labels'],Labels)
            loss.backward()
            optimizer.step()
            
            dice.append(H.DiceScores(sample['labels'],Labels))
            
        print('Training set mean dices: ')
        print(np.mean(dice))
        
        epoch+=1
        UN.eval()
        dices=[]
        losses=[]
        TotalTime+=time.time()-time_check
        time_check=time.time()
        
        
        for i, sample in enumerate(testloader):
            with torch.no_grad():
                torch.cuda.empty_cache()
                result=UN(sample['MRI'])
                dices.append(H.DiceScores(sample['labels'],result))
                losses.append( float(Lossf(sample['labels'],result)))
                specify='_'+sample['timepoint'][0]+'_'+sample['rodent_numer'][0]
                
        
        inftime=(time.time()-time_check)/len(testloader)
        print('Inference time: '+str(inftime)+' per volume')
        
        dices=np.array(dices)
        savedict['dices'].append(dices)
        losses=np.array(losses)
        candidate_loss=np.mean(losses)
        dices=np.mean(dices,axis=0)
        print('Dev set dices:')
        print(dices)
        savedict['losses'].append(losses)
        savedict['mean_dices'].append(dices)
        savedict['mean_loss'].append(candidate_loss)
        savedict['step'].append(step)
        
        
        
        writer.add_scalar('Val Loss',float(candidate_loss), global_step=step)
        writer.add_scalar('Val Dice Mask',float(dices[0]), global_step=step)
        writer.add_scalar('Val Dice Cortex',float(dices[1]), global_step=step)
        writer.add_scalar('Val Dice Hippocampus',float(dices[2]), global_step=step)
        writer.add_scalar('Val Dice Ventricles',float(dices[3]), global_step=step)
        writer.add_scalar('Val Dice Striatum',float(dices[4]), global_step=step)
        writer.add_scalar('Val Mean Dice',float(np.mean(dices)),global_step=step)
        torch.save({
                'epoch': epoch,
                'model_state_dict': UN.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'step': step,
                'bestloss':bestloss,
                'CurrentFold':CurrentFold,
                'bestdice':bestdice,
                'TotalTime':TotalTime,
                'bestlossstep':bestlossstep,
                'savedict':savedict
                }, saveprogress)
        UN.train()

        
        if candidate_loss<bestloss: 
            bestloss=candidate_loss
            bestlossstep=step
            BestStep=step
            bestdice_loss=dices
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': UN.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'step': step,
                    'bestloss':bestloss,
                    'bestdice':bestdice,
                    'dice_scores':dices,
                    'CurrentFold':CurrentFold,
                    'bestlossstep':bestlossstep,
                    'TotalTime':TotalTime,
                    'savedict':savedict
                    }, savefile)
    
    writer.add_graph(UN, [sample['MRI']])
    writer.close()
    
    del UN
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    del optimizer
    CurrentFold+=1


