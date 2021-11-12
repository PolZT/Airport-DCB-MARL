# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 20:01:21 2021

@author: Polatzio
"""
import numpy as np
import matplotlib.pyplot as plt

def process_env_log(input_log_):
    
    self_msg_infos = []
    msg_msg_infos = []
    obs = []
    actions = []
    rw_global = []
    rw_local = []
    mean_msg_info = []
    mean_msg = []
    ground_del_agent = []
    holding_agent = []
    

    input_log__episodes = 1
    steps_ep = len(input_log_)-2

    for e in range(input_log__episodes):

        for s in range(steps_ep):


            k = []

            k.append(e+1)
            k.append(s)
            k = str(k)

            # Agent reward
            rw_loc_st = [input_log_[k]['agents'][i]['rw_agent'] for i in input_log_[k]['agents']]
            rw_local.append(rw_loc_st)
            
            # Agent ground delay
            g_d_step = [input_log_[k]['agents'][i]['ground_delay'] for i in input_log_[k]['agents']]
            ground_del_agent.append(g_d_step)
            # Agent holding
            holding_st = [input_log_[k]['agents'][i]['holding'] for i in input_log_[k]['agents']]
            holding_agent.append(holding_st)
            

            # Agent msg_info
            #self_msg_infos_st = [ float(j) for i in input_log_[k]['agents'] for j in i]
            msg_dim = len(input_log_['[1, 0]']['agents']['0']['msg_info'])
            
            self_msg_infos_st = []
            
            for i in input_log_[k]['agents']:
                self_msg_infos_st_agent = []
                for j in input_log_[k]['agents'][i]['msg_info']:
                    self_msg_infos_st_agent.append(float(j))
                self_msg_infos_st.append(self_msg_infos_st_agent)
            
            #self_msg_infos_st = [[float(input_log_[k]['agents'][i]['msg_info'][0]),float(input_log_[k]['agents'][i]['msg_info'][1])] for i in input_log_[k]['agents'] for j in i]
            self_msg_infos.append(self_msg_infos_st)

            # Mean msg_info
            mean_msg_info.append(float(np.mean(self_msg_infos_st)))

            # Msg msg_infos
            msg_us_st = []
            for u in input_log_[k]['obs']:
                msg_us_st_agent = []
                for i in range(msg_dim):
                    msg_us_st_agent.append(float(u[-i]))
                msg_us_st.append(msg_us_st_agent)
                
            #msg_us_st = [[float(u[-2]), float(u[-1])] for u in input_log_[k]['obs']]
            msg_msg_infos.append(msg_us_st)

            # Mean msg
            mean_msg.append(float(np.mean(msg_us_st)))

            # Actions
            act_st = [a.tolist() for a in input_log_[k]['actions']]

            act_st_ = []
            for lst in act_st:
                lst_new = []
                for i in lst:
                    lst_new.append(float(i))
                act_st_.append(lst_new)

            actions.append(act_st_)

            # Obs
            obs_st = [o.tolist() for o in input_log_[k]['obs']]

            obs_st_ = []
            for lst in obs_st:
                lst_new = []
                for i in lst:
                    lst_new.append(float(i))
                obs_st_.append(lst_new)

            obs.append(obs_st_)
            # Global Reward
            rw_global.append(input_log_[k]['reward'])

    result = {
        'obs': obs,
        'actions': actions,
        'reward': rw_global,
        'msg_msg_info': msg_msg_infos,
        'agent_msg_info': self_msg_infos,
        'agent_reward': rw_local,
        'mean_msg_info': mean_msg_info,
        'mean_msg': mean_msg,
        'agent_ground_delay': ground_del_agent,
        'agent_holding': holding_agent
        }

    return result
