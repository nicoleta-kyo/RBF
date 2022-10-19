# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:26:05 2020

@author: Niki
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from rbf_class_reg import identity, RBF
from sklearn.model_selection import KFold
import numpy as np
from preprocessing import whitener, pca_transform


#------------------ Import the data

data = genfromtxt('training_data.htm', skip_header=27, skip_footer=1)

data_in = data[:,:13]
# drop dummy variable Charles River and var #6
data_in_dropped = data_in[:,[5,10,12]]
data_out = data[:,13]


#----------------- split into train and test
X_train, X_test, y_train, y_test = train_test_split(data_in_dropped, data_out, test_size=0.20)

#----------------- do cross validation
k_folds = 5
num_cv = 5

# grid search
pp = ['norm', 'whiten', 'pca']
k_centers = [10,50,100,150,200,250,300]
width_methods = [None, 'per_cluster_avdist', 'per_cluster_2avdist']
reg = [0,0.3,0.6,0.9]


models_reg = {}
count_model = 0
total = len(pp)*len(k_centers)*len(width_methods)*len(reg)
for pp_v in pp:
    for kcent_v in k_centers:
        for wm_v in width_methods:
            for r_v in reg:
                print('Model '+ pp_v + ', ' + str(kcent_v) + ' centers, width: ' + str(wm_v) + ', reg:' + str(r_v) + ', No: ' +str(count_model)+' out of ' + str(total) + '...')
                model = {}
                model['pp'] = pp_v
                model['k'] = kcent_v
                model['width_method'] = wm_v
                model['reg']=r_v
                errs_cv = np.empty(num_cv)
                for cv_i in range(num_cv):
                    cv = KFold(n_splits=k_folds, shuffle=True, random_state=0)
                    errs_k = np.empty(k_folds)
                    for k_i, (tr, te) in enumerate(cv.split(X_train, y_train)):
                        X_tr_k = X_train[tr]
                        y_tr_k = y_train[tr]
                        X_te_k = X_train[te]
                        y_te_k = y_train[te]
                        
                        #----------------- Pre-processing
                        
                        if pp_v == 'norm':
                            scaler = StandardScaler().fit(X_tr_k)
                            train_in_proc = scaler.transform(X_tr_k)
                            test_in_proc = scaler.transform(X_te_k)                   
                        if pp_v == 'pca':
                            scaler = StandardScaler().fit(X_tr_k)
                            train_in_proc = scaler.transform(X_tr_k)
                            test_in_proc = scaler.transform(X_te_k) 
                            do_pca = pca_transform(train_in_proc,drop=[-1])
                            train_in_proc = do_pca(train_in_proc)
                            test_in_proc = do_pca(test_in_proc)
                        if pp_v == 'whiten':
                            w,v = whitener(X_tr_k)
                            train_in_proc = w(X_tr_k)
                            test_in_proc = w(X_te_k)
    
                        #----------------- Parametrise network
                        
                        rbf = RBF(kcent_v, wm_v, r_v)
                              
                        # --------------- Fit with the training data
                        rbf.fit(train_in_proc, y_tr_k) 
                        
                        # --------------- Evaluate the model
                        errs_k[k_i] = rbf.predict(test_in_proc, y_te_k)
                #        print("Test error for single fold: " + errs_k[k_i])
                    
                    # compute mean error over folds
                    errs_cv[cv_i] = np.mean(errs_k)
    #                print("Mean over k-folds: "+ str(errs_cv[cv_i]))
                mcverr = np.mean(errs_cv)
                print("Mean over n-cv-splits: " + str(mcverr))
                model['err'] = mcverr
                
                models_reg[count_model] = model
                count_model += 1
#
def get_best(x,num_configs):
    # look at data
    modeldata = np.empty((num_configs,2))
    for i in x.keys():
        modeldata[i,0] = i
        modeldata[i,1] = x[i]['err']
        
    modeldata=modeldata[modeldata[:,1].argsort()]
    best_dic = {}
    for i in range(5):
        best_dic[i] = x[modeldata[i,0]]
    return best_dic          


best=get_best(models_reg, total)
for i in best.keys():
    print(best[i])