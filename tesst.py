#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:11:53 2019

@author: riccardo
"""

import helperfile, os

bigstring=[]
for subdir, dirs, files in os.walk("/mnt/90FACB14FACAF60E/CRL3318_HD_MOUSE_DATA"):
    for f in files:
        if '2dseq.nii' == f and 'discarded' not in subdir:
            bigstring.append(os.path.abspath(os.path.join(subdir,f)))
            
size=500
big=0
setlist=[]
helperfile.DEFopt['--overwrite']='True'
helperfile.DEFopt['--multinet']='True'
while big<len(bigstring):
    setlist.append(bigstring[big:big+size])
    big+=size

for s in setlist:
    helperfile.Segment(s)