# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 22:13:45 2021

@author: Polatzio
"""
import numpy as np
from scipy.linalg import sqrtm 


class Messenger: 
    def __init__(self, adjacency_matrix, nodes_features):
        
        self.ad = adjacency_matrix
        self.deg = None
        self.feat = nodes_features
        self.msg = None
        self.act = len(nodes_features)
        
        self.compute_degree()
    
    def compute_msgs(self, nodes_features):
        self.feat = nodes_features
        
        if len(self.feat)==1:
            return [0]
        
        deg_inv_root = np.linalg.inv(sqrtm(self.deg))
        
        A_hat = deg_inv_root @ self.ad @ deg_inv_root
        
        self.msg = A_hat @ self.feat
        
        if self.act != 0 : 
            return self.msg   
        else:
            return np.zeros(self.msg.shape)
        
        
    def compute_degree(self):
        d_ = np.sum(self.ad,axis=1).tolist()
        self.deg = np.eye(self.ad.shape[0], dtype='int') * d_