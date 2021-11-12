import numpy as np
import random as rd
from scipy.linalg import sqrtm 
from numpy import genfromtxt


class Airport:
    
    def __init__(self, rwy_throughput=3, rwy_conf=[0,0,0], msg_dim = 2):
        
        #Runways
        self.rwy_plan = np.array(rwy_conf)
        self.rwy_throughput = np.transpose([rwy_throughput * np.ones(len(rwy_conf), dtype=int)]) #rwy_throughput = operaciones por ciclo (aterrizajes o despegues) 
        self.rwy_change_cost = 0
        
        
        #Capacity        
        self.update_dep_cap()
        self.update_arr_cap()
        
        self.max_cap = rwy_throughput * len(rwy_conf)
        
        #Demand
        self.dep_dem = 0
        self.arr_dem = 0
        self.ground_delay = 0
        self.holding = 0
        
        
        #Balance
        self.dep_bal = 0
        self.arr_bal = 0
        self.CAP_NOISE_FACTOR = 0.2
        self.update_bal_plan()
        
        
        #Message content
        self.msg_dim = msg_dim
        self.msg_info = [0] * self.msg_dim #Cambiar inicialización según dimensiones del mensaje
        
        
        #Is_alive
        self.is_done = False
        self.reward = 0
        
        
        #Reward Factors
        self.SUB_OPT_FACTOR = 0.75
        self.OVERLOAD_FACTOR = 1.5
        self.ARR_FACTOR = 1.1
        self.RWY_CHG_COST_FACTOR = 0.1
        self.GROUND_DELAY_COST_FACTOR = 0.125
        self.HOLDING_COST_FACTOR = 0.25
    
    def update_dep_cap(self):
        dep_rwy = np.where(self.rwy_plan>0, 0, self.rwy_plan)
        self.dep_cap = int(abs(np.dot(dep_rwy,self.rwy_throughput)))

    
    def update_arr_cap(self):
        arr_rwy = np.where(self.rwy_plan<0, 0, self.rwy_plan)
        self.arr_cap = int(np.dot(arr_rwy,self.rwy_throughput))
    

    def update_msg_info(self):

        self.update_bal_plan()
        if self.msg_dim == 1:
            self.msg_info = [(self.dep_dem + self.arr_dem)/self.max_cap]
        elif self.msg_dim == 2:
            self.msg_info = np.array([self.dep_bal,self.arr_bal])
        elif self.msg_dim == 4: #[dep_dem, arr_dem, dep_cap, arr_cap]
            self.msg_info = [self.dep_dem, self.arr_dem, self.dep_cap, self.arr_cap]
    
    def set_demand(self, dep_dem, arr_dem):
        
        if arr_dem:
            self.arr_dem = arr_dem
        if dep_dem:
            self.dep_dem = dep_dem
        
        self.update_msg_info()

        
    def update_dep_bal(self):
        if self.dep_cap == 0:
            dep_cap_ = self.CAP_NOISE_FACTOR
        else: 
            dep_cap_ = self.dep_cap

        self.dep_bal = (self.dep_dem-self.dep_cap)/dep_cap_
    
    def update_arr_bal(self):
        if self.arr_cap == 0:
            arr_cap_ = self.CAP_NOISE_FACTOR
        else: 
            arr_cap_ = self.arr_cap
        
        self.arr_bal = (self.arr_dem-self.arr_cap)/arr_cap_
        
    def update_bal_plan(self):
        self.update_dep_cap()
        self.update_arr_cap()
        self.update_dep_bal()
        self.update_arr_bal()
        
        
    def set_rwy_plan(self, new_rwy_conf):
        if (self.rwy_plan != new_rwy_conf).any():
            self.rwy_change_cost +=1
            
        self.rwy_plan = new_rwy_conf
        self.update_dep_cap()
        self.update_arr_cap()
 
    
    def get_rwy_conf_idx(self, rwy_conf):
        
        if rwy_conf[rwy_conf==0].size > 0:
            inactive_rwy_idx = [i for i in range(len(rwy_conf)) if rwy_conf[i] == 0]
            inactive_rwy_idx  = inactive_rwy_idx[0]
        else:
            inactive_rwy_idx = None
            
        if rwy_conf[rwy_conf==1].size > 0:
            arr_rwy_idx = [i for i in range(len(rwy_conf)) if rwy_conf[i] == 1]
            arr_rwy_idx = arr_rwy_idx[0]
        else:
            arr_rwy_idx = None
        
        if rwy_conf[rwy_conf==-1].size > 0:
            dep_rwy_idx = [i for i in range(len(rwy_conf)) if rwy_conf[i] == -1]
            dep_rwy_idx = dep_rwy_idx[0]
        else:
            dep_rwy_idx = None
 
        return arr_rwy_idx, inactive_rwy_idx, dep_rwy_idx
        
    def adjust_cap(self, rwy_action): #From action of increasing/descreasing/keeping dep. and arr. cap to new rwy conf
        

        new_rwy_conf = self.rwy_plan.copy() #rwy conf to be returned
        arr_rwy_idx, inactive_rwy_idx, dep_rwy_idx = self.get_rwy_conf_idx(new_rwy_conf)
        

        #NOTE: Priority on arrivals
        if rwy_action[1] == 1: #If arr. cap needs to increase
            
            if rwy_action[0] == -1: #if dep cap needs to decrease
                if dep_rwy_idx is not None: #if possible, convert dep rwy to arr rwy
                    new_rwy_conf[dep_rwy_idx] = 1   
                elif inactive_rwy_idx is not None: #if possible, convert inactive rwy to arr rwy
                    new_rwy_conf[inactive_rwy_idx] = 1
                                    
            else: #if dep cap needs to stay (=0) or increase(=1)
                if inactive_rwy_idx is not None: #if possible, convert inactive rwy to arr rwy
                    new_rwy_conf[inactive_rwy_idx] = 1
                elif dep_rwy_idx is not None: #if possible, convert dep rwy to arr rwy
                    new_rwy_conf[dep_rwy_idx] = 1
            
            arr_rwy_idx, inactive_rwy_idx, dep_rwy_idx = self.get_rwy_conf_idx(new_rwy_conf) #update indexes of each rwy conf        
            
            if (rwy_action[0] == 1) and inactive_rwy_idx is not None: #if dep needs to increase and still any inactive
                new_rwy_conf[inactive_rwy_idx] = -1
                # if dep needs to decrease ??
            elif (rwy_action[0] == -1) and dep_rwy_idx is not None:
                new_rwy_conf[dep_rwy_idx] = 0
            
        elif rwy_action[1] == 0: #If arr. cap needs to stay
            # Action for arrival, but nothing. 
            if (rwy_action[0] == 1) and (inactive_rwy_idx is not None): #If dep. cap needs to increase and any inactive rwy
                new_rwy_conf[inactive_rwy_idx] = -1
            elif (rwy_action[0] == -1) and (dep_rwy_idx is not None): #If dep. cap needs to decrease and any dep rwy
                new_rwy_conf[dep_rwy_idx] = 0
            
        elif rwy_action[1] == -1: #If arr. cap needs to decrease
            if (rwy_action[0] == 1): #If dep. cap needs to increase
                if arr_rwy_idx is not None: #If dep. cap needs to increase and any arr rwy
                    new_rwy_conf[arr_rwy_idx] = -1
                elif inactive_rwy_idx is not None:
                    new_rwy_conf[inactive_rwy_idx] = -1
                    
            elif (rwy_action[0] == 0):#If dep. cap needs to stay
                if (arr_rwy_idx is not None): #If dep. cap needs to stay and any arr rwy, set it to inactive
                    new_rwy_conf[arr_rwy_idx] = 0
            
            elif (rwy_action[0] == -1):
                if (arr_rwy_idx is not None): #If dep. cap needs to stay and any arr rwy, set it to inactive
                    new_rwy_conf[arr_rwy_idx] = 0
                if dep_rwy_idx is not None:
                    new_rwy_conf[dep_rwy_idx] = 0
        
        self.set_rwy_plan(new_rwy_conf)
    
    
    def decode_action(self, action):
        
        action[0:2] = np.clip(action[0:2],-1,1) #act on dep_cap and arr_cap
        action[2:4] = np.clip(action[2:4],-1,0) #act on ground_del and holding

        decoded_act = []
        for a in range(2):
            decoded_act.append(int(round(action[a])))
        
        decoded_act.append(action[2])
        decoded_act.append(action[3])
        
        return decoded_act
    
    def decode_action_discrete(self, action):
        
        act_dic = {
            0: [-1, -1, -1, -1], 
            1: [-1, 0, -1, -1],
            2: [-1, 1, -1, -1],
            3: [0, -1, -1, -1],
            4: [0, 0,-1,-1],
            5: [0, 1, -1, -1],
            6: [1, -1, -1, -1],
            7: [1, 0, -1, -1],
            8: [1,1, -1, -1],
            9: [-1,-1,0,0], 
            10: [-1,0,0,0],
            11: [-1,1,0,0],
            12: [0,-1,0,0],
            13: [0,0,0,0],
            14: [0,1,0,0],
            15: [1,-1,0,0],
            16: [1,0,0,0],
            17: [1,1,0,0],
            18: [-1,-1,0,-1], 
            19: [-1,0,0,-1],
            20: [-1,1,0,-1],
            21: [0,-1,0,-1],
            22: [0,0,0,-1],
            23: [0,1,0,-1],
            24: [1,-1,0,-1],
            25: [1,0,0,-1],
            26: [1,1,0,-1],
            27: [-1,-1,-1,0], 
            28: [-1,0,-1,0],
            29: [-1,1,-1,0],
            30: [0,-1,-1,0],
            31: [0,0,-1,0],
            32: [0,1,-1,0],
            33: [1,-1,-1,0],
            34: [1,0,-1,0],
            35: [1,1,-1,0] 
                  }
        
        return act_dic[action[0]]
    
    
    def adjust_dem(self, dem_action, discrete):
        

        if discrete: 
            max_dem_adjust = -1
        else:
            #max_dem_adjust = -round(self.rwy_throughput[0][0])
            max_dem_adjust = -1
        
        self.ground_delay = max(round(self.dep_dem*dem_action[0]),max_dem_adjust )
        self.holding = max(round(self.arr_dem*dem_action[1]), max_dem_adjust)
        
        gd = self.ground_delay
        dp_dm = self.dep_dem
        
        if (self.dep_dem + self.ground_delay) < 0 :
            self.dep_dem = 0 
            self.ground_delay = 0
        else: 
            self.dep_dem += self.ground_delay
        
        
        self.arr_dem = max(self.arr_dem + self.holding, 0)
        # self.dep_dem = max(self.dep_dem + self.ground_delay, 0)
        
        
    
    def perform_action(self, action):
        #Action translation
        if len(action) == 1:
            decoded_action = self.decode_action_discrete(action)
            self.adjust_dem(decoded_action[2:4], discrete=True)
        else:
            decoded_action = self.decode_action(action)
            self.adjust_dem(decoded_action[2:4], discrete=False)

        self.adjust_cap(decoded_action[0:2])
        

        self.update_bal_plan()
        #is alive update??
    
    def evaluate(self):
        
        self.update_bal_plan()
        
        r_arr = min(self.arr_dem, self.arr_cap)
        r_dep = min(self.dep_dem, self.dep_cap)

        # bal > 0 = overload; bal < 0 = underload; bal == 0 -> balance
        

        if self.arr_bal > 0:
            r_arr_bal = ((self.OVERLOAD_FACTOR * self.arr_bal) ** 2) * self.ARR_FACTOR
        else:
            r_arr_bal = abs(self.SUB_OPT_FACTOR * self.arr_bal)
         
        if self.dep_bal > 0:
            r_dep_bal = (self.OVERLOAD_FACTOR * self.dep_bal) **2
        else:
            r_dep_bal = abs(self.SUB_OPT_FACTOR * self.dep_bal)
        

        r_rwy_change = self.rwy_change_cost * self.RWY_CHG_COST_FACTOR

        r_ground_delay = self.ground_delay * self.GROUND_DELAY_COST_FACTOR

        r_holding = self.holding * self.HOLDING_COST_FACTOR

        self.reward = round(np.float(r_arr + r_dep - r_arr_bal - r_dep_bal - r_rwy_change + r_ground_delay + r_holding), 3) 

        self.rwy_change_cost = 0
        
        
        return self.reward
    
