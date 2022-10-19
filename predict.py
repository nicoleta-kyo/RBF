# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:44:09 2020

@author: Niki
"""

from numpy import genfromtxt

preddata = genfromtxt('prediction_data.htm', skip_header = 6, skip_footer=1)

preddata = preddata[:,[0,1,2,4,5,6,7,8,9,10,11,12]]


Phi = rbf._calc_Phi(preddata_pre)

preds = np.dot(Phi,rbf.W)
