# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:16:08 2020

@author: Niki
"""


import numpy as np
from numpy import linalg as la
from sklearn import cluster
from scipy.spatial import distance
import pdb


# network
class RBF():
    
    def __init__(self, n_centers, width_method):
        self.n_centers = n_centers
        self.width_method = width_method
    
    def _set_centres(self,x):
        k_means = cluster.KMeans(n_clusters = self.n_centers)
        k_means.fit(x)
        self.centers = k_means.cluster_centers_
        self.labels = k_means.labels_
    
    def fit(self,x, y):
        self._set_centres(x)
        self._calc_width(x)
        Phi = self._calc_Phi(x)
        self.W = np.dot(la.pinv(Phi),y)
    
    def predict(self, x, y):
        Phi = self._calc_Phi(x)
        preds = np.dot(Phi,self.W)
        err = mse(preds, y)
        return err
    
    def _calc_width(self,x):
        # calculate sigma and Phi matrix
        if self.width_method == 'single':
            # calculate average distance between centres
            self.width = 2*np.mean(distance.pdist(self.centers))
        if self.width_method == 'per_cluster_avdist':
            # calculate average distance between the data points and the centre
            self.width=np.empty(self.n_centers)
            for c in range(self.n_centers):
                 cl = x[self.labels== c,:]
                 d_ave = np.mean(distance.cdist(cl,self.centers[c,:].reshape(1,-1)))
                 self.width[c] = d_ave
        if self.width_method == 'per_cluster_2avdist':
            # calculate average distance between the data points and the centre
            self.width=np.empty(self.n_centers)
            for c in range(self.n_centers):
                 cl = x[self.labels== c,:]
                 d_ave = np.mean(distance.cdist(cl,self.centers[c,:].reshape(1,-1)))
                 self.width[c] = 2*d_ave
        if self.width_method == 'per_cluster_std':
            # calculate average distance between the data points and the centre
            self.width=np.empty(self.n_centers)
            for c in range(self.n_centers):
                 cl = x[self.labels== c,:]
                 self.width[c] = np.std(cl)   
        if self.width_method == 'cov':
            self.width = np.empty((self.n_centers,x.shape[1],x.shape[1]))
            for c in range(self.n_centers):
                cl = x[self.labels== c,:]
                cov = np.cov(cl, rowvar=False)
                if cov.ndim == 0:
                    pdb.set_trace()
                try:
                    inv_cov = la.inv(cov)
                except:
                    inv_cov = la.pinv(cov)
                self.width[c] = inv_cov 
    
    def _calc_Phi(self,x):
        if self.width_method is None:
            Phi = distance.cdist(x,self.centers)
        # calculate sigma and Phi matrix
        if self.width_method == 'single':
            dists = distance.cdist(x,self.centers)
            Phi = np.exp((-1*dists)/2*(self.width**2))
        if self.width_method == 'per_cluster_avdist':
            # calculate Phi using each center's sigma
            dists = distance.cdist(x,self.centers)
            Phi = np.empty(dists.shape)
            for c in range(self.n_centers):
                Phi[:,c] = np.exp((-1*dists[:,c])/2*(self.width[c]**2))
        if self.width_method == 'per_cluster_2avdist':
            # calculate Phi using each center's sigma
            dists = distance.cdist(x,self.centers)
            Phi = np.empty(dists.shape)
            for c in range(self.n_centers):
                Phi[:,c] = np.exp((-1*dists[:,c])/2*(self.width[c]**2))
        if self.width_method == 'per_cluster_std':
            dists = distance.cdist(x,self.centers)
            Phi = np.empty(dists.shape)
            for c in range(self.n_centers):
                Phi[:,c] = np.exp((-1*dists[:,c])/2*(self.width[c]**2))    
        if self.width_method == 'cov':
            # calculate cov matrix for each cluster
            Phi = np.empty((x.shape[0],self.n_centers))
            for c in range(self.n_centers):
                for dp in range(x.shape[0]):
                    diff=x[dp,:]-self.centers[c]
                    exp_term = -0.5*np.dot(np.dot(diff.T,self.width[c]),diff)
                    # put upper boundary so as to not result in overflow
                    exp_term = 700 if exp_term > 700 else exp_term
                    Phi[dp,c] = np.exp(exp_term)
        
        # add bias center
        Phi = np.hstack([np.ones((x.shape[0],1)),Phi])
        return Phi
         


# calculate error
def mse(pred, t):
    err = np.sum((pred-t)**2)/2
#    print(err)
    return err

# identity function
def identity(x):
    return x


        

