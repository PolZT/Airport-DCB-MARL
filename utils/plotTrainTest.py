# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 15:31:15 2021

@author: Polatzio
"""
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from IPython.display import IFrame
import plotly
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import json as js
import pandas as pd


def plot_train_test(train, test, log, exe_params, n=100, no_plot=False, agent = 0, t = 'NoTime', folder='./plots/'):

    
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(30,20))
    
    total_train = len(train.history['episode_reward'])
    total_test = len(test.history['episode_reward'])
    
    y_vals1 = []
    y_vals2 = []

    n_layers = exe_params["NUM_LAYERS"]
    u_layer = exe_params["NUM_UNITS_PER_LAYER"]
    
    selfish_idx = exe_params["SELFISH"]
    msg = exe_params["MESSAGES_ACTIVE"]

    fig.suptitle(f'Model {agent} | messaging = {msg} |selfish = {selfish_idx} | {n_layers} layers and {u_layer} units per layer', fontsize=32)

    dct = {
        'dep_dem': 0,
        'arr_dem': 1,
        'dep_cap': 2,
        'arr_cap': 3,
        'msg': 4
        }
    
    steps = range(len(log['obs']))
    
    if len(log['actions'][0][agent])>1:
        dep_act = [log['actions'][i][agent][0] for i in steps]
        arr_act = [log['actions'][i][agent][1] for i in steps]
        gd_act = [log['actions'][i][agent][2] for i in steps]
        hd_act = [log['actions'][i][agent][3] for i in steps]
    
    
    g_delay = [-log['agent_ground_delay'][i][agent] for i in steps]
    holding = [-log['agent_holding'][i][agent] for i in steps]
    
    dep_dem_ = [log['obs'][i][agent][dct['dep_dem']] for i in steps]
    dep_dem = [dd-gd  for dd , gd in zip(dep_dem_,g_delay)]
    
    arr_dem_ = [log['obs'][i][agent][dct['arr_dem']] for i in steps]
    arr_dem = [ad-hd  for ad , hd in zip(arr_dem_,holding)]
    
    dep_cap = [log['obs'][i][agent][dct['dep_cap']] for i in steps]
    dep_cap = dep_cap[1:]
    dep_cap.append(dep_cap[-1])
    
    arr_cap = [log['obs'][i][agent][dct['arr_cap']] for i in steps]
    arr_cap = arr_cap[1:]
    arr_cap.append(arr_cap[-1])
    

    
    dem_tot =  [sum(i) for i in zip(dep_dem, arr_dem)]
    dem_gd = [sum(i) for i in zip(g_delay, dem_tot)]
    
    cap_tot = [sum(i) for i in zip(dep_cap,arr_cap)]
    
    n_agents = len(log['agent_reward'][0])
    
    mean_rw = [(log['reward'][i])/n_agents for i in steps]
    rw_agent = [log['agent_reward'][i][agent] for i in steps]
    
    x_vals_train = []
    x_vals_test = []
    
    if total_train <= n:
        n = total_train
        p = 1
    else:
        p = int(total_train/n)
    
    for i in range(n):
        p_ = p*i
        x_vals_train.append(p_)
        y_vals1.append(train.history['episode_reward'][p_])

    
    mean1 = np.mean(np.array(y_vals1))
    ax1.axhline(y = mean1, color = 'm', linestyle = '--', alpha=0.4)
    ax1.scatter(x_vals_train,y_vals1, s=5)
    ax1.plot(x_vals_train,y_vals1,color ='cyan', alpha=0.5)
    
    # ax3 = dep
    
    ax3.bar(steps, dep_dem, alpha=0.7, color= 'dodgerblue')
    ax3.bar(steps, g_delay, bottom=dep_dem, color = 'cyan', alpha=0.5)
    ax3.plot(steps, dep_cap, linewidth=2.5, color = 'k', alpha=0.7, marker='_')
    if len(log['actions'][0][agent])>1:
        ax3.plot(steps, dep_act, linewidth=2.5, color = 'b', alpha=1, marker='o', linestyle = '--')
        ax3.plot(steps, gd_act, linewidth=2.5, color = 'cyan', alpha=1, marker='x', linestyle = '--')
    
    # ax combined arr dep
    ax5.bar(steps, dep_dem, alpha=0.7, color='dodgerblue')
    ax5.bar(steps, arr_dem, bottom=dep_dem, color = 'g', alpha=0.7)
    ax5.bar(steps, g_delay, bottom=dem_tot, color = 'cyan', alpha=0.5)
    ax5.bar(steps, holding, bottom=dem_gd, color = 'lime', alpha=0.5)
    ax5.plot(steps, cap_tot, linewidth=2.5 ,color = 'k', marker='_')
    # ax5.plot(steps, mean_rw, color='r', label='mean reward')
    # ax5.plot(steps, rw_agent, color='g', label='agent reward')

    
    ax1.set_ylabel('Train: episode_reward')
    ax1.set_title('Train: episode_reward')
    ax1.legend('episode_reward', loc='lower right')
    
    ax3.set_ylabel('Test log: departure demand vs capacity')
    ax3.set_xlabel('Test log: step')
    ax3.set_title('Test log: departure demand vs capacity per step')
    
    
    ax5.set_ylabel('Test log: combined demand vs capacity')
    ax3.set_xlabel('Test log: step')
    ax5.set_title('Test log: combined demand vs capacity per step')
    
        
    if total_test <= n:
        n = total_test
        p = 1
    else:
        p = int(total_test/n)
        

    for i in range(n):
        p_ = p*i
        x_vals_test.append(p_)
        y_vals2.append(test.history['episode_reward'][p_])



    mean2 = np.mean(np.array(y_vals2))
    ax2.axhline(y = mean2, color = 'm', linestyle = '--', alpha=0.4)
    ax2.scatter(x_vals_test,y_vals2, s=5)
    ax2.plot(x_vals_test, y_vals2, color ='cyan', alpha=0.5)
    
    #ax4 = arr
    ax4.bar(steps, arr_dem, color = 'g', alpha=0.7)
    ax4.bar(steps, holding, bottom=arr_dem, color = 'lime', alpha=0.5)
    ax4.plot(steps, arr_cap, linewidth=2.5,color = 'k', alpha=0.7, marker='_')
    if len(log['actions'][0][agent])>1:
        ax4.plot(steps, arr_act, linewidth=2.5,color = 'g', alpha=1, marker='o', linestyle = '--')
        ax4.plot(steps, hd_act, linewidth=2.5, color = 'lime', alpha=1, marker='x', linestyle = '--')

    ax4.set_ylabel('Test log: arrival demand vs capacity')
    ax4.set_xlabel('Test log: step')
    ax4.set_title('Test log: arrival demand vs capacity per step')
    
    ax2.set_ylabel('Test: episode_reward')
    ax2.set_title('Test: episode_reward')
    ax2.legend('episode_reward', loc='lower right')
    
    #ax 6 rw per step
    ax6.plot(steps, mean_rw, color='b', label='mean reward')
    ax6.plot(steps, rw_agent, color='g', label='agent reward')
    ax6.set_ylabel('Test log: reward')
    ax6.set_xlabel('Test log: step')
    ax6.set_title('Test log: reward per step')
    
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)

    fig.show()
    PLOT_FILE = folder+f'/plots/{t} Model {agent} , messaging-{msg} , selfish-{selfish_idx}, {n_layers} layers and {u_layer} units per layer.png'
    fig.savefig(PLOT_FILE) 
    
    

def compute_means(test):
    ep_rw = np.array(test.history['episode_reward'])
    n_steps_per_ep = np.array(test.history['nb_steps'])
    rw_per_step = ep_rw/n_steps_per_ep
    
    m_ep_rw = np.mean(ep_rw)
    m_n_steps_per_ep = np.mean(n_steps_per_ep)
    m_rw_per_step = np.mean(rw_per_step)
    
    return m_rw_per_step, m_n_steps_per_ep, m_ep_rw



def plot_rand_params(rand_param_search):
    x_vals = [rand_param_search[i][1]['NUM_UNITS_PER_LAYER'] for i in range(len(rand_param_search))]
    c_vals = [rand_param_search[i][1]['SELFISH'] for i in range(len(rand_param_search))]
    z_vals = [rand_param_search[i][0] for i in range(len(rand_param_search))]
    m_vals = [rand_param_search[i][1]['MESSAGES_ACTIVE'] for i in range(len(rand_param_search))]
    y_vals = [rand_param_search[i][1]['NUM_LAYERS'] for i in range(len(rand_param_search))]
    

    x = np.array(x_vals[:10])
    m = np.array(m_vals[:10])
    c = np.array(c_vals[:10])
    z = np.array(z_vals[:10])
    y = np.array(y_vals[:10])
    
    m_symbol = ['diamond' if i==True  else 'x' for i in m]
    
    df = pd.DataFrame()
    
    df['NUM_UNITS_PER_LAYER'] = x
    df['SELFISH'] = c
    df['REWARD'] = z
    df['MESSAGES_ACTIVE'] = m
    df['NUM_LAYERS'] = y
    
    fig1 = px.scatter_3d(
        df,
        x='NUM_UNITS_PER_LAYER',
        y='NUM_LAYERS',
        z='REWARD',
        color = 'SELFISH',
        symbol='MESSAGES_ACTIVE'
        )

    #Make Plot.ly Layout
    fig1.update_layout(
        showlegend = True,
        legend = dict(
            xanchor="right",
            orientation="h",
            yanchor="bottom",
            
            )
    )

    fig1.write_html("./myplot.html")

