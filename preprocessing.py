# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:22:54 2020

@author: Niki
"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def whitener(data):
  sigma = np.cov(data, rowvar=False)
  mu = np.mean(data, axis=0)
  values, vectors = LA.eig(sigma)
  l = np.diag(values ** -0.5)
  
  def func(datapoints):
    return np.array( [ l.dot(vectors.T.dot(d - mu)) for d in datapoints ])
  
  def inv_func(datapoints):
    return np.array( [ LA.inv(vectors.T).dot(LA.inv(l).dot(d)) + mu for d in datapoints ] )
  
  return func, inv_func

def pca_transform(data, drop = None):
  mu = np.mean(data, axis=0)
  sigma = np.cov(data, rowvar=False)
  values, vectors = LA.eig(sigma)
  
  components = sorted( zip(values, vectors.T), key = lambda vv: vv[0], reverse=True )
  if drop[0] != -1:
      for i in drop:
          components.pop(i)

  def func(datapoints):
    result = []
    for d in datapoints:
      t = [ vector.dot(d - mu) for (value, vector) in components ]
      result.append(t)
    return np.array(result)
  
  return func