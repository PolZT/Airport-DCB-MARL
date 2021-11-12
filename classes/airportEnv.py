import gym
from gym import Env, spaces 
from gym.spaces import Box
import numpy as np
from airport import Airport
from messenger import  Messenger
from trafficGenerator import TrafficGenerator


class AirportEnv(Env):
    
    def __init__(self, n_agents=2, max_dem = 2, rand_graph = False, msg_act = 2, log_act = False, graph = None, is_discrete = False):
        
        self.log_act = log_act
        #Setup agents
        self.n_agents = n_agents
        self.airports = []
        self.max_dem = max_dem
        self.msg_dim = msg_act
        
        self.obs_per_agent = 4 + self.msg_dim
        if is_discrete: 
            self.act_per_agent = 1
        else:
            self.act_per_agent = 4
        
        self.graph = graph
        
        self.total_ground_delays = 0
        self.total_holdings = 0
        
        #Adjacency Matrix
        if rand_graph and self.graph is None:
            self.adjacency = np.random.randint(0,2, size=(self.n_agents,self.n_agents)) + np.random.randint(0,2, size=(self.n_agents,self.n_agents))
            np.clip(self.adjacency,0,1, out=self.adjacency)
            self.adjacency = (self.adjacency+self.adjacency.T)/2
            self.adjacency = np.array(self.adjacency, dtype='int')
            np.fill_diagonal(self.adjacency, 1)
        elif (not rand_graph) and (self.graph is None):
            self.adjacency = np.ones((self.n_agents,self.n_agents))
        
        elif self.graph is not None:
            assert len(self.graph) == self.n_agents
            self.adjacency = self.graph
            np.fill_diagonal(self.adjacency, 1)
            
        np.fill_diagonal(self.adjacency,0)
        
        #Traffic Generation
        self.traffic_gen = TrafficGenerator(self.max_dem, self.n_agents, np.zeros(self.n_agents), np.zeros(self.n_agents), self.adjacency)
        self.traffic = self.traffic_gen.generate_traffic(np.zeros(self.n_agents,dtype='int'), np.zeros(self.n_agents,dtype='int'))
        
        #From Traffic matrix to demands per airport
        if self.n_agents > 1:
            dep_dems = np.sum(self.traffic, axis=1)
            arr_dems = np.sum(self.traffic, axis=0)
        else:
            dep_dems = np.array([self.traffic[0]])
            arr_dems = np.array([self.traffic[1]])
        
        #Initialize observation and action spaces range (low and high)
        obs_low = np.array([])
        obs_high = np.array([])
        
        act_low = np.array([])
        act_high = np.array([])
        
        for i in range(self.n_agents): 
            self.airports.append(Airport(msg_dim =self.msg_dim))
            obs_low = np.concatenate([obs_low, np.array([0]*self.obs_per_agent)])
            obs_high = np.concatenate([obs_high, np.array([12]*self.obs_per_agent)])    
            act_low = np.concatenate([act_low, np.array([-1, -1, -1, -1])])
            act_high = np.concatenate([act_high, np.array([1, 1, 0, 0])])
            self.airports[i].set_demand(dep_dems[i],arr_dems[i])
            
        self.messenger = Messenger(self.adjacency, np.zeros(self.msg_dim))
        
        #Actions [dep_cap_act, arr_cap_act, ground_del, holding] increase = 1, decrease = -1 or stay = 0, -1= holding/ground_delay or 0 = nothing
        self.action_space = Box(low= act_low , high = act_high, dtype=np.float16)
        
        #Observations [dep_dem, arr_dem, dep_cap, arr_cap, msg1, dep_dem2, arr_dem2, dep_cap2, arr_cap2, msg2...]
        self.observation_space = Box(obs_low, obs_high, dtype=np.float16)

        #Initialize environment log
        #len_log = self.observation_space.shape[0] + self.action_space.shape[0]
        #self.env_log = np.zeros()
        if self.log_act:
            self.iter_counter = [0,-1]
            self.env_log ={
                "env_params": {
                    "n_agents": self.n_agents,
                    "max_dem": self.max_dem,
                    "obs_per_agent": self.obs_per_agent,
                    "act_per_agent": self.act_per_agent,
                    "adjacency_mtx": self.adjacency,
                    }
                
                }
        
    def step(self,actions):
        
        reward_global = 0
        rewards_per_agent = []
        next_state = np.zeros((self.n_agents, self.obs_per_agent))
        done = False
        

        actions = np.array(actions)
        actions_mtx = actions.reshape((self.n_agents,self.act_per_agent))
        
        
        msg_infos = []           
            
        ground_delays = np.zeros(self.n_agents)
        holdings = np.zeros(self.n_agents)
        
        for a in range(self.n_agents):

            #perform action 
            action = actions_mtx[a]
            
            self.airports[a].perform_action(action)
            
            #get action in agent
            ground_delays[a] = self.airports[a].ground_delay
            holdings[a] = self.airports[a].holding
        
        self.traffic , ground_delays_ = self.traffic_gen.update_traffic(self.traffic, ground_delays, holdings)
        
        
        if self.n_agents > 1:
            dep_dems = np.sum(self.traffic, axis=1)
            arr_dems = np.sum(self.traffic, axis=0) + holdings #holdings is always <= 0
            
            
        else:
            dep_dems = np.array([self.traffic[0]])
            arr_dems = np.array([self.traffic[1]]) + holdings
        
        for a in range(self.n_agents):
            
            if arr_dems[a] < 0:
                arr_dems[a] = 0
                holdings[a] = 0

            self.airports[a].holding = holdings[a]
                
            self.airports[a].ground_delay = ground_delays_[a]
                
            self.airports[a].set_demand(dep_dems[a], arr_dems[a])
            
            #reward    
            rw_agent = self.airports[a].evaluate()
            reward_global += rw_agent 
            rewards_per_agent.append(rw_agent)
            
        
        self.total_ground_delays -= sum(ground_delays)
        self.total_holdings -= sum(holdings)
        
        #Preparing next state
        self.traffic = self.traffic_gen.generate_traffic(ground_delays, holdings)
        
        if self.n_agents > 1:
            dep_dems = np.sum(self.traffic, axis=1)
            arr_dems = np.sum(self.traffic, axis=0) - holdings #holdings is always <= 0
        else:
            dep_dems = np.array([self.traffic[0]])
            arr_dems = np.array([self.traffic[1]])
        
        
        
        for a in range(self.n_agents):    
            #observe (return next state)
            if self.log_act:
                self.env_log[str(self.iter_counter)]["agents"][str(a)] ={
                    "msg_info": self.airports[a].msg_info, 
                    "rw_agent" : rewards_per_agent[a],
                    'ground_delay': self.airports[a].ground_delay,
                    'holding': self.airports[a].holding}
            
            msg_infos.append(self.airports[a].msg_info)
            next_state[a,2]=self.airports[a].dep_cap
            next_state[a,3]=self.airports[a].arr_cap
            
            self.airports[a].set_demand(dep_dems[a], arr_dems[a])
             
            next_state[a,0] = dep_dems[a]
            next_state[a,1] = arr_dems[a]
            
            # is done
            if self.airports[a].is_done:
                done = True
                
           
        messages = self.messenger.compute_msgs(msg_infos)
        self.msgs = messages
        next_state[:,4:] = messages
        
        next_state_ = [np.float16(i) for i in next_state]
        next_state = tuple(next_state_)

        #any other info
        if self.log_act:
            self.env_log[str(self.iter_counter)]["actions"] = actions_mtx
            self.env_log[str(self.iter_counter)]["reward"] = reward_global 
            self.iter_counter[1] += 1
            self.env_log[str(self.iter_counter)] = {"agents":{}}
            self.env_log[str(self.iter_counter)]["traffic"] = self.traffic
            self.env_log[str(self.iter_counter)]["obs"] = next_state
            
        reward = (reward_global, rewards_per_agent)
        
        return next_state, reward, done, {}


    def render(self, mode='human'):
        
        #print(f'{self.msgs}')
        pass
    
    def reset(self):
        
        if self.log_act:       
            self.iter_counter[0] += 1
            self.iter_counter[1] = 0


        
        self.airports = []
        for _ in range(self.n_agents): 
            self.airports.append(Airport(msg_dim =self.msg_dim))
        
        next_state = np.zeros((self.n_agents, self.obs_per_agent))
              
        
        self.traffic = self.traffic_gen.generate_traffic(np.zeros(self.n_agents,dtype='int'), np.zeros(self.n_agents,dtype='int'))

        if self.log_act:
            self.env_log[str(self.iter_counter)] ={"traffic": self.traffic, "agents":{}}
        
        if self.n_agents > 1:
            dep_dems = np.sum(self.traffic, axis=1)
            arr_dems = np.sum(self.traffic, axis=0)
        else:
            dep_dems = np.array([self.traffic[0]])
            arr_dems = np.array([self.traffic[1]])
            
        msg_infos = []
        
        #observe (return next state)
        for a in range(self.n_agents):
            
            if self.log_act:
                self.env_log[str(self.iter_counter)]["agents"][str(a)] ={"msg_info": self.airports[a].msg_info, 'ground_delay': self.airports[a].ground_delay, 'holding': self.airports[a].holding}
                
            msg_infos.append(self.airports[a].msg_info)
            
            
            next_state[a,2]=self.airports[a].dep_cap
            next_state[a,3]=self.airports[a].arr_cap
            
            self.airports[a].set_demand(dep_dems[a], arr_dems[a])
             
            next_state[a,0] = dep_dems[a]
            next_state[a,1] = arr_dems[a]
            
        messages = self.messenger.compute_msgs(msg_infos)
        next_state[:,4:] = messages
        
        next_state_ = [np.float16(i) for i in next_state]
        next_state = tuple(next_state_)
        
        if self.log_act:
            self.env_log[str(self.iter_counter)]["obs"] = next_state
        
        return next_state
    
    def return_env_log(self):
        if self.log_act:
            return self.env_log
        else:
            return {}
        
    def get_delays(self):
        return (self.total_ground_delays, self.total_holdings)
    
    def reset_delays(self):
        self.total_ground_delays = 0
        self.total_holdings = 0
