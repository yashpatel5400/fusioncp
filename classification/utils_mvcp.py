#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:32:10 2024

@author: eochoa
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, LinearRegression


from sklearn.utils.fixes import parse_version, sp_version

# This is line is to avoid incompatibility if older SciPy version.
# You should use `solver="highs"` with recent version of SciPy.
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

from sklearn.linear_model import QuantileRegressor


def gram_schmidt(V):
    """
    Orthonormalizes a set of vectors using the Gram-Schmidt process.
    
    Parameters:
        V (numpy.ndarray): Input vectors as columns of a matrix.
        
    Returns:
        numpy.ndarray: Orthonormalized vectors as columns of a matrix.
    """
    vec_dim, num_vecs  = V.shape
    Q = np.zeros((vec_dim, num_vecs))
    for i in range(num_vecs):
        # Orthogonalize current vector against previous vectors
        q = V[:,i]
        if i > 0:
            for j in range(i):
                q -= np.dot(Q[:,j], V[:,i]) * Q[:,j]
        # Normalize the resulting vector
        Q[:,i] = q / np.linalg.norm(q)
    return Q

def project_u(s, th):
    u = np.array([np.cos(th),np.sin(th)])#[:,None]
    U = np.array([[-1.],
                  [-1.]])
    #U = np.concatenate([u, np.eye(2)[:,1:]],1)
    U = np.concatenate([u[:,None], U],1)
    orthonormalized = gram_schmidt(U)
    v = U[:,1]
    
    s_u = np.transpose(np.array([s[0], s[1]]),(1, 2, 0)) @ u
    s_u_orth = np.transpose(np.array([s[0], s[1]]),(1, 2, 0)) @ v

    return s_u, s_u_orth

def project_u_mv(s, u):
    s_u = np.transpose(np.array(s),(1, 2, 0)) @ u
    return s_u, None

def simulate_s_work(N, # sample size
               d, #dimension
               L, # n of labels
               scale, 
               K, # n of views
               N_train, 
               seed,
               adapt=True):
    
    np.random.seed(seed)
    X = 2*np.random.rand(N,d)-1
    theta = scale*(2*(np.random.rand(d, L))-1)
    P = softmax(X @ theta, axis=1)
    y = np.array([np.random.choice(L, size=1, replace=True, p=P[i,:])[0] for i in range(N)])

    X_train = X[:N_train,:]
    y_train = y[:N_train]

    X_test = X[N_train:,:]
    y_test = y[N_train:]
    
    scores_list = []
    acc_list = []
    dim_list = np.int64(np.ones(K)/K * d)
    dim_list[0] = dim_list[0] + d - dim_list.sum()
    dim_list = dim_list.cumsum()
    dim_list = np.concatenate([np.array([0]), dim_list])
    
    for i in range(K):
        clf_ = LogisticRegression(multi_class='multinomial')
        clf_ = clf_.fit(X_train[:,dim_list[i]:dim_list[i+1]], y_train)
        
        f = clf_.predict_proba(X_test[:,dim_list[i]:dim_list[i+1]])
        if adapt:
            cal_pi = f.argsort(1)[:,::-1]
            cal_srt = np.take_along_axis(f,cal_pi,axis=1)
            s = np.array([cal_srt.cumsum(axis=1)[i,cal_pi.argsort(1)[i]] for i in range(f.shape[0])])
        else:
            s = f
        acc = clf_.score(X_test[:,dim_list[i]:dim_list[i+1]], y_test)
        scores_list.append(s)
        acc_list.append(acc)
        
    clf_ = LogisticRegression(multi_class='multinomial')
    clf_ = clf_.fit(X_train, y_train)
    s_full = clf_.predict_proba(X_test)
    scores_list.append(s_full)
    
    return scores_list, y_train, y_test, np.array(acc_list) 

def gen_u(K, n_th=400, seed=0):
    np.random.seed(seed=seed)
    U_list = []
    for i in range(1, K):
        U = np.abs(np.random.multivariate_normal(np.zeros(i+1), np.eye(i+1), size=n_th))
        U = np.array([u/np.linalg.norm(u) for u in U])
        U_list.append(U)
    return U_list

def simulate_s(N, # sample size
               d, #dimension
               L, # n of labels
               scale, 
               corr,
               var,
               K, # n of views
               N_train, 
               seed,
               prop=None,
               adapt=True):
    np.random.seed(seed)
    X = 2*np.random.rand(N,d)-1
    #X = np.random.multivariate_normal(np.zeros(d),cov=var*(np.eye(d) + corr*np.ones((d,d))),size=N)
    theta = scale*(2*(np.random.rand(d, L))-1)
    P = softmax(X @ theta, axis=1)
    y = np.array([np.random.choice(L, size=1, replace=True, p=P[i,:])[0] for i in range(N)])

    X_train = X[:N_train,:]
    y_train = y[:N_train]

    X_test = X[N_train:,:]
    y_test = y[N_train:]
    
    f_list = []
    scores_list = []
    acc_list = []
    if prop is None:
        dim_list = np.int64(np.ones(K)/K * d)
    else:
        dim_list = np.int64(prop * d)
        
    dim_list[0] = dim_list[0] + d - dim_list.sum()
    dim_list = dim_list.cumsum()
    dim_list = np.concatenate([np.array([0]), dim_list])
    
    for i in range(K):
        clf_ = LogisticRegression(multi_class='multinomial')
        clf_ = clf_.fit(X_train[:,dim_list[i]:dim_list[i+1]], y_train)
        #s = clf_.predict_proba(X_test[:,dim_list[i]:dim_list[i+1]])
        
        f = clf_.predict_proba(X_test[:,dim_list[i]:dim_list[i+1]])
        f_list.append(f)
        if adapt:
            cal_pi = f.argsort(1)[:,::-1]
            cal_srt = np.take_along_axis(f,cal_pi,axis=1)
            s = np.array([cal_srt.cumsum(axis=1)[i,cal_pi.argsort(1)[i]] for i in range(f.shape[0])])
        else:
            s = f
            
        acc = clf_.score(X_test[:,dim_list[i]:dim_list[i+1]], y_test)
        scores_list.append(s)
        acc_list.append(acc)
    
    f_pf = np.mean(f_list,0) #f_list
    cal_pi = f_pf.argsort(1)[:,::-1]
    cal_srt = np.take_along_axis(f_pf,cal_pi,axis=1)
    s_pre_fused = np.array([cal_srt.cumsum(axis=1)[i,cal_pi.argsort(1)[i]] for i in range(f_pf.shape[0])])
    

    clf_ = LogisticRegression(multi_class='multinomial')
    clf_ = clf_.fit(X_train, y_train)
    f_full = clf_.predict_proba(X_test)
    cal_pi = f_full.argsort(1)[:,::-1]
    cal_srt = np.take_along_axis(f_full,cal_pi,axis=1)
    s_full = np.array([cal_srt.cumsum(axis=1)[i,cal_pi.argsort(1)[i]] for i in range(f_full.shape[0])])
    
    scores_list.append(s_full)
    
    return scores_list, s_pre_fused, y_train, y_test, np.array(acc_list) 

# function to split data into calibration and teste sets
def split_cal_test(smx, labels_test, cal_prop=0.5, seed=0):
    
    n = int(smx.shape[0] * cal_prop) # number of calibration points
    idx = np.array([1] * n + [0] * (smx.shape[0]-n)) > 0
    np.random.seed(seed)
    np.random.shuffle(idx)
    cal_smx, val_smx = smx[idx,:], smx[~idx,:]
    cal_labels, val_labels = labels_test[idx], labels_test[~idx]

    return cal_smx, val_smx, cal_labels, val_labels, idx




def conform_prediction_adp(cal_smx, val_smx, cal_labels, n, alpha, val_labels=None, cond=0):
    
    # 1: get conformal scores. n = calib_Y.shape[0]
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    
    #cal_scores = cal_smx[np.arange(n),cal_labels]
    # 2: get adjusted quantile
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)

    #n_list = [cal_scores[cal_labels==i].shape[0] for i in range(cal_smx.shape[1])]
    # 3: get prediction sets
    #prediction_sets = qhat >= val_smx # 3: form prediction sets
    
    if val_labels is None: 
        return prediction_sets, qhat
    # Calculate empirical coverage
    empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    return (prediction_sets, empirical_coverage), qhat

def conform_prediction(cal_smx, val_smx, cal_labels, n, alpha, val_labels=None, cond=0):
    
    # 1: get conformal scores. n = calib_Y.shape[0]
    
    cal_scores = cal_smx[np.arange(n),cal_labels]
    # 2: get adjusted quantile
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(cal_scores, q_level, method='higher')
     
    # 3: form prediction sets
    prediction_sets = qhat >= val_smx 
    
    if val_labels is None: 
        return prediction_sets, qhat
    # Calculate empirical coverage
    empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    return (prediction_sets, empirical_coverage), qhat


def calibrate_beta(alpha, 
                   cal_smx1, 
                   cal_smx2, 
                   cal_labels, 
                   n_th=20, 
                   max_itr=10):
    beta = alpha
    diff = 3
    change = .1
    n = cal_smx1.shape[0]
    i = 0
    while abs(diff)>1 and i<max_itr:
        beta_prev = beta
        cov_list = []
        q_list = []
        for th in np.linspace(0,np.pi/2,n_th):

            cal_s_u, _ = project_u([cal_smx1, cal_smx2], th)
            cal_scores = cal_s_u[np.arange(n),cal_labels]
            q_level = np.ceil((n+1)*(1-beta))/n
            qhat = np.quantile(cal_scores, q_level, method='higher')
            cov_list.append(cal_scores <= qhat)
            q_list.append(qhat)
            
        diff = np.ceil((n+1)*(1-alpha)) - np.prod(cov_list,0).sum()
        if np.prod(cov_list,0).sum() < np.ceil((n+1)*(1-alpha)):
            beta = beta_prev - change / 2
        else:
            beta = beta_prev + change / 2
        change = abs(beta - beta_prev)
        i = i + 1
    return beta_prev, np.array(q_list)

def mvcp(quant_smx1, 
         quant_smx2, 
         quant_labels, cal_smx1, 
         val_smx1, 
         cal_smx2, 
         val_smx2, 
         cal_labels, 
         alpha, 
         val_labels, 
         n_th=20):
    
    beta, q_array = calibrate_beta(alpha, quant_smx1, quant_smx2, quant_labels, n_th)
    prediction_sets_list = []
    n = cal_smx1.shape[0]
    th_s = np.linspace(0,np.pi/2,n_th) 
    t = np.max([project_u([cal_smx1, cal_smx2], th)[0][np.arange(n),cal_labels]/q for th, q in zip(th_s, q_array)], 0)
    q_level = np.ceil((n+1)*(1-alpha))/n
    t_quant = np.quantile(t, q_level, method='higher')
    
    for j, th in enumerate(np.linspace(0,np.pi/2,n_th)):

        val_s_u, val_s_u_orth = project_u([val_smx1, val_smx2], th)
        
        prediction_sets = val_s_u <= q_array[j] * t_quant
        
        prediction_sets_list.append(prediction_sets)
        
    prediction_sets_ = np.prod(prediction_sets_list,0) == 1
    empirical_coverage = prediction_sets_[np.arange(prediction_sets_.shape[0]),val_labels].mean()
        
    return prediction_sets_, empirical_coverage

def calibrate_beta_mv(alpha, 
                   score_list, 
                   cal_labels, 
                   n_th=400,
                   U=None,
                   max_itr=10):
    beta = alpha
    diff = 3
    change = .1
    M = len(score_list)
    n = cal_labels.shape[0]
    i = 0
    if U is None:
        U = np.abs(np.random.multivariate_normal(np.zeros(M), np.eye(M), size=n_th))
    else:
        n_th=U.shape[0]
        
    while abs(diff)>1 and i<max_itr:
        beta_prev = beta
        cov_list = []
        q_list = []
        for th in range(n_th):#np.linspace(0,np.pi/2,n_th):
            u = U[th,:] #/ np.linalg.norm(U[th,:])
            cal_s_u, _ = project_u_mv(score_list, u)
            cal_scores = cal_s_u[np.arange(n),cal_labels]
            q_level = np.ceil((n+1)*(1-beta))/n
            qhat = np.quantile(cal_scores, q_level, method='higher')
            cov_list.append(cal_scores <= qhat)
            q_list.append(qhat)
            
        diff = np.ceil((n+1)*(1-alpha)) - np.prod(cov_list,0).sum()
        if np.prod(cov_list,0).sum() < np.ceil((n+1)*(1-alpha)):
            beta = beta_prev - change / 2
        else:
            beta = beta_prev + change / 2
        change = abs(beta - beta_prev)
        i = i + 1
    return beta_prev, np.array(q_list), U

def mvcp_2(quant_scores_list, 
           quant_labels, 
           cal_scores_list, 
           cal_labels, 
           val_scores_list, 
           alpha, 
           val_labels, 
          n_th=400,
          U=None):
    
    beta, q_array, U = calibrate_beta_mv(alpha, quant_scores_list, quant_labels, n_th, U)
    n_th = U.shape[0]
    prediction_sets_list = []
    n = cal_labels.shape[0]
    #th_s = np.linspace(0,np.pi/2,n_th) 
    t = np.max([project_u_mv(cal_scores_list, U[th,:])[0][np.arange(n),cal_labels]/q for th, q in enumerate(q_array)], 0)
    q_level = np.ceil((n+1)*(1-alpha))/n
    t_quant = np.quantile(t, q_level, method='higher')
    
    for j, th in enumerate(np.linspace(0,np.pi/2,n_th)):

        val_s_u, val_s_u_orth = project_u_mv(val_scores_list, U[j,:])
        
        prediction_sets = val_s_u <= q_array[j] * t_quant
        
        prediction_sets_list.append(prediction_sets)
        
    prediction_sets_ = np.prod(prediction_sets_list,0) == 1
    empirical_coverage = prediction_sets_[np.arange(prediction_sets_.shape[0]),val_labels].mean()
        
    return prediction_sets_, empirical_coverage