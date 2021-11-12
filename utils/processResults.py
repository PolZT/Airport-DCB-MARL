# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 22:45:18 2021

@author: Polatzio
"""
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# %%

def get_stats(ls):
    n = len(ls)
    if n < 1:
        maximum, minimum, mean, std_dev = 0 , 0, 0 ,0
    else:
        mean = sum(ls) / n
        minimum = min(ls)
        maximum = max(ls)
        var = sum((x - mean)**2 for x in ls) / n
        std_dev = var ** 0.5
    return maximum, minimum, mean, std_dev
    

def process_results(Y_LIMIT):
    result_files = []
    MSG_range = [0,1,2,4]
    for MSGS_ACT in MSG_range:
        
        # Number of agents
        n_agents_range = [x for x in  range(1,4)]
        n_agents_range.reverse()
        #n_agents_range = [2,3]
        for N_AGENTS in n_agents_range:
            
            # Choose type of model DQN (DISCRETE=True) or DDPG (DISCRETE=False)
            DISCRETE_range = [False,True]
            for DISCRETE in DISCRETE_range:
                if DISCRETE:
                    alg = 'DQN'
                else:
                    alg = 'DDPG'
    
                # Choose the local vs global reward weights (only local matters: SELFISH = 1)
                # (only global reward matters: SELFISH = 0)            

                SELFISH_range = [0, 0.25, 0.5, 0.75, 1]
                #SELFISH_range = [0.0]
                for SELFISH in SELFISH_range:
                    

                    try:    
                        EXPERIMENT_FOLDER = f'./results/{alg}/{N_AGENTS}-{alg}-{SELFISH}-{MSGS_ACT}'
    
                        json_files = [pos_json for pos_json in os.listdir(EXPERIMENT_FOLDER+'/') if pos_json.endswith('.json')]
                    except:
                        SELFISH=float(SELFISH)
                        EXPERIMENT_FOLDER = f'./results/{alg}/{N_AGENTS}-{alg}-{SELFISH}-{MSGS_ACT}'
                        json_files = [pos_json for pos_json in os.listdir(EXPERIMENT_FOLDER+'/') if pos_json.endswith('.json')]
                        
                        
                    finally:
                        result_files.append((N_AGENTS, alg, SELFISH, MSGS_ACT, EXPERIMENT_FOLDER, EXPERIMENT_FOLDER+'/'+json_files[0]))
    
    data = {
        'id':[],
        'n_agents': [],
        'algorithm': [],
        'selfish': [],
        'msg': [],
        'rws': [],
        'rws_agent':[],
        'rw_max': [],
        'rw_min': [],
        'rw_mean': [],
        'rw_std_dev': [],
        'gds':[],
        'gd_max': [],
        'gd_min': [],
        'gd_mean': [],
        'gd_std_dev': [],
        'holds':[],
        'hold_max': [],
        'hold_min': [],
        'hold_mean': [],
        'hold_std_dev': []    
        }
    data_raw = {
    'id':[],
    'n_agents': [],
    'algorithm': [],
    'selfish': [],
    'msg': [],
    'rws': [],
    'rws_agent':[],
    'gds':[],
    'holds':[],
    }
    
    Z_SCORE=-1.25

    
    for N_AGENTS, alg, SELFISH, MSGS_ACT, EXPERIMENT_FOLDER, FILE_NAME in result_files:
        
        print(FILE_NAME)
        f = open(FILE_NAME,)
        
        ID_ = EXPERIMENT_FOLDER.split(sep='/')
        ID = ID_[-1]
        # returns JSON object as 
        # a dictionary
        results = json.load(f)
        results = results[:-5]
        
        rw_list_ = [res[0] for res in results]
        gd_list_ = [res[2][0] for res in results]
        hold_list_ = [res[2][1] for res in results]
        
        z_score_ = stats.zscore(rw_list_)
        z_score_ = [i for i in rw_list_ if i >= Y_LIMIT]
 
        #z_score = z_score_[z_score_>Z_SCORE]
        keep_idxs = len(z_score_)
        #keep_idxs = 0
        
        rw_list = rw_list_[:keep_idxs]
        rw_list_norm_ag = [r/N_AGENTS for r in rw_list]
        
        gd_list = gd_list_[:keep_idxs]
        hold_list = hold_list_[:keep_idxs]
        
        
        for i in range(keep_idxs):
            data_raw['id'].append(ID)
            data_raw['n_agents'].append(N_AGENTS)
            data_raw['algorithm'].append(alg)
            data_raw['selfish'].append(SELFISH)
            data_raw['msg'].append(MSGS_ACT)
            data_raw['rws'].append(rw_list[i])
            data_raw['rws_agent'].append(rw_list_norm_ag[i])
            data_raw['gds'].append(gd_list[i])
            data_raw['holds'].append(hold_list[i])

        
        rw_max, rw_min, rw_mean, rw_std_dev = get_stats(rw_list_norm_ag)
        gd_max, gd_min, gd_mean, gd_std_dev = get_stats(gd_list)
        hold_max, hold_min, hold_mean, hold_std_dev = get_stats(hold_list)
    

        data['id'].append(ID)
        data['n_agents'].append(N_AGENTS)
        data['algorithm'].append(alg)
        data['selfish'].append(SELFISH)
        data['msg'].append(MSGS_ACT)
        data['rws'].append(rw_list)
        data['rws_agent'].append(rw_list_norm_ag)
        data['rw_max'].append(rw_max)
        data['rw_min'].append(rw_min)
        data['rw_mean'].append(rw_mean)
        data['rw_std_dev'].append(rw_std_dev)
        data['gds'].append(gd_list)
        data['gd_max'].append(gd_max)
        data['gd_min'].append(gd_min)
        data['gd_mean'].append(gd_mean)
        data['gd_std_dev'].append(gd_std_dev)
        data['holds'].append(hold_list)
        data['hold_max'].append(hold_max)
        data['hold_min'].append(hold_min)
        data['hold_mean'].append(hold_mean)
        data['hold_std_dev'].append(hold_std_dev)
        
    
        
    df_ = pd.DataFrame(data)
    df_raw = pd.DataFrame(data_raw)
    return df_, df_raw



# %%

df_results, df_raw = process_results(0)
 # %%
sns.set_theme(style="darkgrid")
# a2 = df_raw[(df_raw['algorithm'] == 'DDPG') & (df_raw['n_agents'] == 1)]
a2 = df_raw[(df_raw['algorithm'] == 'DQN')]
a1 = df_raw[(df_raw['algorithm'] == 'DDPG')]
f = sns.catplot(x="n_agents", y="rws_agent", hue="selfish", 
           data=df_raw, kind='box', col='algorithm')
#%%
# g = sns.kdeplot(x="gds", y="holds", 
#             data=df_raw ,  legend=True, kde=True, hue='algorithm')

# f = sns.displot(x="gds", y="holds", 
#             data=df_raw , kind='kde', legend=True, hue='algorithm')
# sns.set_theme(style="darkgrid")
g= sns.jointplot(
    data=df_raw,
    x="gds", y="holds", hue="algorithm",
    kind="kde", fill=False, )
# g= sns.jointplot(
#     data=df_raw,
#     x="gds", y="rws_agent", hue="algorithm",
#     kind="kde", fill=False, )
# g= sns.jointplot(
#     data=df_raw,
#     x="holds", y="rws_agent", hue="algorithm",
#     kind="kde", fill=False, )
# %%
v = sns.violinplot(data=df_raw, x="msg", y="gds", hue="algorithm", split=True,  inner="quartile")

# %%
sns.set_palette('deep')
g = sns.catplot(x="selfish", y="rws_agent", hue="algorithm",
                col="algorithm",
                data=a2, kind="box");

#g.color_palette("PuBuGn_d")
# g= sns.jointplot(
#     data=a2,
#     x="gds", y="n_agents", hue="algorithm",
#     kind="kde", color='#FFA500')

# g= sns.jointplot(
#     data=a2,
#     x="holds", y="n_agents", hue="algorithm",
#     kind="kde")
 
# %%
sns.set_palette('deep')
z = sns.PairGrid(df_raw[['algorithm','selfish', 'msg','n_agents','gds','holds', 'rws_agent']], hue='algorithm')
z.map_upper(sns.scatterplot)
z.map_lower(sns.kdeplot, fill=False, common_norm=True)
z.map_diag(sns.histplot, kde=True)


# %%

# f1 = sns.catplot(x="selfish", y="rws_agent",
#                 hue="msg", 
#                 data=a1, kind='box', col='n_agents', legend=True)
#f1.set(ylim=(Y_LIMIT,500))

# a2 = df_raw[(df_raw['algorithm'] == 'DDPG') & (df_raw['n_agents'] == 1)]
#a2 = df_raw[(df_raw['algorithm'] == 'DQN')]
# for Y_LIMIT in [-1e6, -4e5, -1e4, -1e3, -1e2, 0]:
#     df_results, df_raw = process_results(Y_LIMIT)
#     for a in ['DQN','DDPG']:
#         a2 = df_raw[(df_raw['algorithm'] == a)]
#         f = sns.catplot(x="selfish", y="rws_agent", hue="msg", 
#                     data=a2, kind='box', col='n_agents', legend=True)

    
        
#         f.savefig(f'./results/plots/{a}/all-{a}-selfish-msg-{Y_LIMIT}.png')
            

#f.set(ylim=(Y_LIMIT, 500))
# f = sns.stripplot(x="selfish", y="rws_agent",
#                 hue="msg",
#                 data=a1, marker='o', alpha=0.5)
# %%
sns.set_palette("PuBuGn_d")
#a2 = df_raw[(df_raw['n_agents'] == n) & (df_raw['algorithm'] == a)]

#sns.set_theme(style="darkgrid")
#ax = sns.lineplot(data=a2, x='selfish',y='rws_agent', hue="msg", style='msg')
# %%
#df_results, df_raw = process_results(Y_LIMIT)

def process_train():
    
    
    data = {
    'id':[],
    'n_agents': [],
    'algorithm': [],
    'selfish': [],
    'msg': [],
    'iter':[],
    'nb_steps':[],
    'ep_rw_train': []
    }
    

    MSG_range = [0,1,2,4]
    for MSGS_ACT in MSG_range:
        
        # Number of agents
        n_agents_range = [x for x in  range(1,4)]
        n_agents_range.reverse()
        #n_agents_range = [2,3]
        for N_AGENTS in n_agents_range:
            
            # Choose type of model DQN (DISCRETE=True) or DDPG (DISCRETE=False)
            DISCRETE_range = [False,True]
            for DISCRETE in DISCRETE_range:
    
                # Choose the local vs global reward weights (only local matters: SELFISH = 1)
                # (only global reward matters: SELFISH = 0)            
                SELFISH_range = [x/100 for x in  range(0,100,25)]
                SELFISH_range.reverse()
                
                #SELFISH_range = [0.0]
                for SELFISH in SELFISH_range:
                    
                    if DISCRETE:
                        alg = 'DQN'
                    else:
                        alg = 'DDPG'
                        
                    EXPERIMENT_FOLDER = f'./results/{alg}/{N_AGENTS}-{alg}-{SELFISH}-{MSGS_ACT}/logs'
                    ID_ = EXPERIMENT_FOLDER.split(sep='/')
                    ID = ID_[-2]
                    
                    
                    nb_steps = []
                    ep_rw_train =[]
                    for path, _, files in os.walk(EXPERIMENT_FOLDER):
                        
                        for n, file in enumerate(files):
                            if 'train_log' in file:
                                f = open(EXPERIMENT_FOLDER+'/'+file,)
                                train_log = json.load(f)
                                
                                nb_steps.extend(train_log['nb_steps'])
                                ep_rw_train.extend(train_log['episode_rewardtest'])
                    
                    nb_steps
                                
                                

#process_train()
                                
                                
                    
                      
            
            