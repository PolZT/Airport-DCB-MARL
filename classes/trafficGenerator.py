# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 22:15:13 2021

@author: Polatzio
"""
from scipy.linalg import sqrtm 
import numpy as np
import random as rd

class TrafficGenerator: 
    
    def __init__(self, max_dem, n_agents, ground_delays, holdings, adjacency_matrix):
        
        self.max_dem = max_dem
        self.n_agents = n_agents
        self.ad_mtx = adjacency_matrix
        self.ground_delays = [np.int32(_) for _ in ground_delays]
        self.holdings = [np.int32(_) for _ in holdings]
        self.traffic = np.zeros(shape=(self.n_agents,self.n_agents))
        self.idxs_gd = None #np.zeros(self.n_agents, dtype=int)

    def generate_traffic(self, ground_delays, holdings):
        
        self.ground_delays = [int(_) for _ in ground_delays]
        self.holdings = [int(_) for _ in holdings]
            
        
        traffic = np.random.randint(0,self.max_dem+1,size=(self.n_agents,self.n_agents))
        

        if self.n_agents > 1:            
            np.fill_diagonal(traffic, 0)
    
            #Update traffic with the holdings and delays comming from the previous step
            for j in range(self.n_agents):            
                
                if self.idxs_gd:
                    idxs_gd = self.idxs_gd[j]
                    for i in idxs_gd:
                        traffic[j,i] += 1 
    
            traffic = traffic * self.ad_mtx
        
        else:
            traffic = np.random.randint(0, self.max_dem+1, size=2)

            traffic[0] -= self.ground_delays
            traffic[1] -= self.holdings
        
        self.traffic = traffic 
        return self.traffic
    
    def update_traffic(self, traffic, gd, hold):
        #Update traffic with the holdings and delays comming from the current step
        self.ground_delays = [int(_) for _ in gd]
        self.holdings = [int(_) for _ in hold]
        actual_gd = (np.zeros(len(self.ground_delays))).tolist()
        idxs_gd = []
        
        if self.n_agents == 1:
            
            if (traffic[0] + self.ground_delays) < 0:
                traffic[0] = 0 
                self.ground_delays = [0]
            else:
                traffic[0] += self.ground_delays

            return traffic, self.ground_delays
        else:
            for j in range(self.n_agents):            
                
                choices_pool = np.where(traffic[j,:] != 0) [0]
                choices_pool = choices_pool.tolist()
                
                
                if choices_pool:
                    idxs_gd.append(rd.choices(choices_pool, k=int(-self.ground_delays[j])))
                else:
                    idxs_gd.append([])
                
                for i in idxs_gd[j]:
                    traffic[j,i] -= 1
                    actual_gd[j] -=1

            self.idxs_gd = idxs_gd
            actual_gd_ = [int(_) for _ in actual_gd]
            return traffic, actual_gd
        
        
        
