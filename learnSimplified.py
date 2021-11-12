# Modified from keras-rl ddpg-pendulum.py example
#%%Imports
import numpy as np
import gym
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2

from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from rl.agents import DDPGAgent,  DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess

from airportEnv import AirportEnv 
from multiAgent import MultiAgent


import time
import plotTrainTest  as pTT
import envLogToolkit as logTool
import json

# %% Static params






ENV_NAME = 'AirportEnv'

# Parameters
N_AGENTS = 3
VERBOSE = 1
MAX_STEPS_EPISODE = 100
N_EPISODES_TEST = 10

MAX_DEM_REF = 4

MAX_DEM = max(int(MAX_DEM_REF / max(N_AGENTS-1,1)), 1) 

# Enable comunication between agents [0 = No com, 1 = usage, 2 = balance, 4 = observations]
MSGS_ACT = 1

# Chose type of model DQN (DISCRETE=True) or DDPG (DISCRETE=False)
DISCRETE = True

if DISCRETE:
    alg = 'DQN'
else:
    alg = 'DDPG'

# Chose the local vs global reward weights (only local matters: SELFISH = 1)
# (only global reward matters: SELFISH = 0)
SELFISH =  0.5


# If RAND_GRAPH = False and GRAPH = None, a fully conected graph will be created

GRAPH_FILE_NAME = None

if GRAPH_FILE_NAME:
    GRAPH = np.genfromtxt(GRAPH_FILE_NAME, delimiter=',')
    RAND_GRAPH = False
else:
    GRAPH = None
    
RAND_GRAPH = False

rand_param_search = [
    (float('-inf'), {"DISCRETE": -1, "NUM_LAYERS": 0, 
                     "NUM_UNITS_PER_LAYER": 0, 
                     "SELFISH": -1, 
                     'MESSAGES_ACTIVE': None}) ] * 1

EXPERIMENT_FOLDER = f'./results/{alg}/{N_AGENTS}-{alg}-{SELFISH}-{MSGS_ACT}-test'

#create experiment folder and subfolders
if not os.path.isdir(EXPERIMENT_FOLDER):
    os.mkdir(EXPERIMENT_FOLDER) #F
    os.mkdir(EXPERIMENT_FOLDER+'/w') #F
    os.mkdir(EXPERIMENT_FOLDER+'/plots') #F
    os.mkdir(EXPERIMENT_FOLDER+'/logs') #F


ITER_OPS = 1 #len(SELFISH_range)
# %% Opt loop
for iterator in range(ITER_OPS):

    if DISCRETE :
        N_ITERATIONS_TRAIN = int(10e3)
        NUM_LAYERS =  5 
        LAYER_SIZE =  1100  
    else:
        N_ITERATIONS_TRAIN = int(5e3)
        # Layer params
        NUM_LAYERS =  8 
        LAYER_SIZE =  300 
    
    N_LOG_INTERVAL = int(N_ITERATIONS_TRAIN/5)
        
    # %%% iteration params
    
    # PARAMETERS
    # DQN and DDPG Params 
    SEED = [np.random.randint(1, 100) for _ in range(25)] #F 

    BATCH_SIZE = 128 # np.random.randint(32, 128)
    LEARNING_RATE = 1e-3
    TARGET_MODEL_UPDATE = 1e-3 #np.random.choice([1e-2, 1e-3])
    # Number of steps before the backward method starts the update
    N_WARMUP = 500 # BATCH_SIZE*2
    MEMORY_SIZE = int(1e4)# np.random.randint(1e6, 1e7)
    MEMORY_WINDOW = 1
    CLIPNORM =  1e-4 # np.random.choice([1e-2, 1e-3, 1e-4]) #0.01 # np.random.uniform(0.5, 1.)

    
    # DDPG Specific (random process to explore/expoit the env)
    MU=0.
    SIGMA=0.3
    SIGMA_MIN=0.01
    N_STEPS_ANNEALING = 200
    GAMMA= .999
    
    
    # Execution dictionary, params initialization
    exe_params = {
        "N_AGENTS": N_AGENTS,
        "SEED": SEED,
        "MEMORY_SIZE": MEMORY_SIZE,
        "MEMORY_WINDOW": MEMORY_WINDOW,
        "NUM_UNITS_PER_LAYER": LAYER_SIZE,
        "NUM_LAYERS": NUM_LAYERS,
        "BATCH_SIZE": BATCH_SIZE,
        "N_WARMUP": N_WARMUP,
        "N_ITERATIONS_TRAIN": N_ITERATIONS_TRAIN,
        "N_LOG_INTERVAL": N_LOG_INTERVAL,
        "LEARNING_RATE": LEARNING_RATE,
        "MAX_STEPS_EPISODE": MAX_STEPS_EPISODE,
        "N_EPISODES_TEST": N_EPISODES_TEST,
        "MAX_DEM": MAX_DEM,
        "MU": MU,
        "SIGMA": SIGMA,
        "SIGMA_MIN": SIGMA_MIN,
        "N_STEPS_ANNEALING": N_STEPS_ANNEALING,
        "GAMMA": GAMMA,
        "TARGET_MODEL_UPDATE": TARGET_MODEL_UPDATE,
        "CLIPNORM": CLIPNORM,
        "MESSAGES_ACTIVE": MSGS_ACT,
        "RAND_GRAPH": RAND_GRAPH,
        "GRAPH_FILE_NAME": GRAPH_FILE_NAME, 
        "DISCRETE": DISCRETE,
        "SELFISH": SELFISH
        }
    # Environment and seeds intialization
    
    env = AirportEnv(n_agents=N_AGENTS, max_dem=MAX_DEM, msg_act=MSGS_ACT, #F
                     rand_graph=RAND_GRAPH, graph=GRAPH, is_discrete = DISCRETE) #F
    env.seed(SEED[1]) #F
    n_actions_per_agent = env.act_per_agent
    n_obs_per_agent = env.obs_per_agent
    model_architecture = [LAYER_SIZE for i in range(NUM_LAYERS)]

    # %%% Build agents

    def build_actor(model_architecture_):

        actor = Sequential(name='actor')
        actor.add(Flatten(input_shape=(MEMORY_WINDOW, n_obs_per_agent))) #F
        for layer, u in enumerate(model_architecture_):
            initializer = tf.keras.initializers.GlorotUniform(seed=SEED[layer]) 
            actor.add(Dense(units=u, kernel_initializer=initializer,kernel_regularizer=l2(0.01))) #F
            actor.add(Activation('relu')) #F


        actor.add(Dense(n_actions_per_agent, kernel_initializer=initializer)) #F
        actor.add(Activation('tanh')) #F

        return actor

    def build_actor2(model_architecture_):

        observation_input = Input(shape=(MEMORY_WINDOW, n_obs_per_agent),
                                  name='actor_observation_input')

            
        x = Flatten()(observation_input)
        for layer, u in enumerate(model_architecture_):
            initializer = tf.keras.initializers.GlorotUniform(seed=SEED[layer])
            x = Dense(units=u, kernel_initializer=initializer)(x)
            x = Activation('relu')(x)
            
            #x = BatchNormalization()(x)

        x = Dense(n_actions_per_agent, kernel_initializer=initializer)(x)
        x = Activation('tanh')(x)
        actor = Model(inputs=observation_input, outputs=x, name='actor')
        return actor

    def build_critic(model_architecture_):

        action_input = Input(shape=(n_actions_per_agent,), name='action_input')
        observation_input = Input(shape=(MEMORY_WINDOW, n_obs_per_agent),
                                  name='critic_observation_input')

        flattened_observation = Flatten()(observation_input)
        
        for lay, u in enumerate(model_architecture_):
            initializer = tf.keras.initializers.GlorotUniform(seed=SEED[-lay])
            if lay == 0: 
                x = Dense(units=u, kernel_initializer=initializer)(flattened_observation)
            else:         
                x = Dense(units=u, kernel_initializer=initializer)(x)
            x = Activation('relu')(x)

            if lay == 2 :#int(len(model_architecture_)/2):
                x = Concatenate()([action_input, x])
                

        x = Dense(1, kernel_initializer=initializer)(x)
        critic = Model(inputs=[action_input, observation_input],
                       outputs=x, name='critic')

        return critic, action_input
    
    def build_critic2(model_architecture_):

        action_input = Input(shape=(n_actions_per_agent,), name='action_input')
        observation_input = Input(shape=(MEMORY_WINDOW, n_obs_per_agent),
                                  name='critic_observation_input')

        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])

        for lay, u in enumerate(model_architecture_):
            initializer = tf.keras.initializers.GlorotUniform(seed=SEED[-lay])
            x = Dense(units=u, kernel_initializer=initializer)(x)
            x = Activation('relu')(x)

        x = Dense(1, kernel_initializer=initializer)(x)
        critic = Model(inputs=[action_input, observation_input],
                       outputs=x, name='critic')

        return critic, action_input
    
    
    def build_dqn_model(model_architecture_):        
        dqn_net = Sequential(name='dqn_net')
        dqn_net.add(Flatten(input_shape=(MEMORY_WINDOW, n_obs_per_agent)))
        for layer, u in enumerate(model_architecture_):
            initializer = tf.keras.initializers.GlorotUniform(seed=SEED[layer])
            dqn_net.add(Dense(units=u, kernel_initializer=initializer))
            dqn_net.add(Activation('relu'))
            dqn_net.add(BatchNormalization())
        dqn_net.add(Dense(units=36, kernel_initializer=initializer))
        
        return dqn_net
    

    tf.keras.backend.clear_session() #F
    
    memory = SequentialMemory(limit=MEMORY_SIZE, window_length=MEMORY_WINDOW)
    
    if not DISCRETE: 
        actor = build_actor2(model_architecture) #F
    
        critic, action_input = build_critic2(model_architecture) #F
     
        # print(actor.summary())
        # print(critic.summary())
    
        
        random_process = GaussianWhiteNoiseProcess(mu=MU, sigma=SIGMA,
                                                   sigma_min=SIGMA_MIN,
                                                   n_steps_annealing=N_STEPS_ANNEALING,
                                                   size=n_actions_per_agent)
    
        #random_process = OrnsteinUhlenbeckProcess(size=n_actions_per_agent, theta=.15, mu=0., sigma=0.99)
        agent = DDPGAgent(nb_actions=n_actions_per_agent, actor=actor,
                        critic=critic, critic_action_input=action_input,
                        memory=memory, nb_steps_warmup_critic=N_WARMUP,
                        nb_steps_warmup_actor=N_WARMUP,
                        random_process=random_process, gamma=GAMMA,
                        target_model_update=TARGET_MODEL_UPDATE,
                        batch_size=BATCH_SIZE)
    else:
        
        dqn_net = build_dqn_model(model_architecture) #F
        
        policy = BoltzmannQPolicy()
        agent =  DQNAgent(model=dqn_net, nb_actions=36, memory=memory, nb_steps_warmup=N_WARMUP,
                          target_model_update=TARGET_MODEL_UPDATE, policy=policy, enable_double_dqn=True)

    agents_list = []
    opts = []
    for i in range(N_AGENTS):
        agents_list.append(agent)#F
        opts.append(Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM)) #F

    agent = MultiAgent(agents_list, selfish=SELFISH, n_obs_per_agent = n_obs_per_agent) 

    agent.compile(opts, metrics=['mae']) #F

    # %%% Train

    train = agent.fit(env, nb_steps=N_ITERATIONS_TRAIN, #F
                      log_interval=N_LOG_INTERVAL, visualize=False, #F
                      verbose=VERBOSE, nb_max_episode_steps=MAX_STEPS_EPISODE) #F

    # %%% Test
    
    env.reset_delays()
    # Finally, evaluate our algorithm.
    test = agent.test(env, nb_episodes=N_EPISODES_TEST, visualize=True, #F
                      verbose=VERBOSE, nb_max_episode_steps=MAX_STEPS_EPISODE) #F

    delays = env.get_delays() #F
    
    delays_per_step_agent = tuple(d/(N_EPISODES_TEST*N_AGENTS*MAX_STEPS_EPISODE) for d in delays)
    
    env.close() #F

    mean_q_step, mean_steps, mean_ep_rw = pTT.compute_means(test=test) #F
    # %%% Save history
    t = time.localtime() #F
    timestamp = time.strftime('%y-%m-%d_%H.%M.%S', t) #F

    FILE_NAME = EXPERIMENT_FOLDER + '/w/'+alg+'_agents-'+str(N_AGENTS)+'_sfh-'+str(SELFISH)+'_msg-'+str(MSGS_ACT)+'_'+timestamp+'_weights.h5f'

    agent.save_weights(FILE_NAME, overwrite=True) #F

    train_dict = {}
    train_dict['nb_steps'] = [int(i) for i in train.history['nb_steps']]
    train_dict['episode_rewardtest'] = [float(i) for i in train.history['episode_reward']]
    train_dict['nb_episode_steps'] = [int(i) for i in train.history['nb_episode_steps']]
    
    FILE_NAME_TRAIN = EXPERIMENT_FOLDER+'/logs/train_log_'+timestamp+'.json'
    with open(FILE_NAME_TRAIN, 'w') as convert_file:
        convert_file.write(json.dumps(train_dict, indent=3)) #F
    
    test_dict = {}
    test_dict['nb_steps'] = [int(i) for i in test.history['nb_steps']]
    test_dict['episode_reward'] = [float(i) for i in test.history['episode_reward']]

    FILE_NAME_TEST =  EXPERIMENT_FOLDER+'/logs/test_log_'+timestamp+'.json'
    with open(FILE_NAME_TEST, 'w') as convert_file:
        convert_file.write(json.dumps(test_dict, indent=3)) #F
        

    # %%% Prints
    exe_params["MEAN_Q_STEP"] = mean_q_step
    exe_params["MEAN_STEPS"] = mean_steps
    exe_params["MEAN_EP_RW"] = mean_ep_rw

    rand_param_search.append((exe_params["MEAN_EP_RW"], exe_params, delays_per_step_agent)) #F
    rand_param_search.sort(reverse=True) #F

    # %%Write file
    FILE_NAME_PARAMS = EXPERIMENT_FOLDER+'/'+str(rand_param_search[0][1]["MEAN_EP_RW"])+'_'+timestamp+'.json'
    # FILE_NAME_PARAMS = 'ops_exe_param_noche_25-07-21_v2.json'
    with open(FILE_NAME_PARAMS, 'w') as convert_file:
        convert_file.write(json.dumps(rand_param_search, indent=3)) #F


# %% Test for log

    test_env = AirportEnv(n_agents=N_AGENTS, max_dem = MAX_DEM, msg_act = MSGS_ACT, rand_graph = RAND_GRAPH, log_act = True, graph = GRAPH, is_discrete=DISCRETE) #F
    
    test_env.seed(SEED) #F
    
    tst = agent.test(test_env, nb_episodes=1, visualize=True, #F
                      verbose=VERBOSE, nb_max_episode_steps=20) #F
        
    env_log = test_env.return_env_log() #F
    
    log = logTool.process_env_log(env_log) #F
    
# %%
    for a in range(N_AGENTS):
        pTT.plot_train_test(train = train, test = test, log = log, n=int(1e6), exe_params = exe_params, t= timestamp, agent = a, folder = EXPERIMENT_FOLDER) #F
      
# %%
    FILE_NAME_LOG = EXPERIMENT_FOLDER+'/logs/demo_'+timestamp+'.json'
    
    with open(FILE_NAME_LOG, 'w') as convert_file:
        convert_file.write(json.dumps(log, indent=3)) #F

    #del agent

