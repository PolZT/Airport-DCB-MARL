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

# Enable comunication between agents [0 = No com, 1 = usage, 2 = balance, 4 = observations]
for MSGS_ACT in [0,1,2,4]:
    
    # Number of agents
    n_agents_range = [x for x in  range(1,4)]
    n_agents_range.reverse()
    n_agents_range = [1,2,3]
    for N_AGENTS in n_agents_range:
        
        # Choose type of model DQN (DISCRETE=True) or DDPG (DISCRETE=False)
        for DISCRETE in [False, True]:
            
            # Choose the local vs global reward weights (only local matters: SELFISH = 1)
            # (only global reward matters: SELFISH = 0)            
            SELFISH_range = [x/100 for x in  range(0,101,50)]
            SELFISH_range.reverse()
            SELFISH_range = [0.25, 0.75]
            for SELFISH in SELFISH_range:
                
                rand_param_search = [
                    (float('-inf'), {"DISCRETE": -1, "NUM_LAYERS": 0, 
                                     "NUM_UNITS_PER_LAYER": 0, 
                                     "SELFISH": -1, 
                                     'MESSAGES_ACTIVE': None}) ] * 5
            
                ENV_NAME = 'AirportEnv'

                VERBOSE = 1
                
                MAX_STEPS_EPISODE = 100
                N_EPISODES_TEST = 10
                
                MAX_DEM_REF = 4
                
                MAX_DEM = max(int(MAX_DEM_REF / max(N_AGENTS-1,1)), 1) 
                
                if DISCRETE:
                    alg = 'DQN'
                else:
                    alg = 'DDPG'
                
                # If RAND_GRAPH = False and GRAPH = None, a fully conected graph will be created
                GRAPH_FILE_NAME = None
                
                if GRAPH_FILE_NAME:
                    GRAPH = np.genfromtxt(GRAPH_FILE_NAME, delimiter=',')
                    RAND_GRAPH = False
                else:
                    GRAPH = None
                    
                RAND_GRAPH = False
                
                t = time.localtime()
                timestamp = time.strftime('%H:%M:%S', t)
                
                ITER_OPS = 5 
                
                EXPERIMENT_FOLDER = f'./results/{alg}/{N_AGENTS}-{alg}-{SELFISH}-{MSGS_ACT}'
                
                
                #create experiment folder and subfolders
                if not os.path.isdir(EXPERIMENT_FOLDER):
                    os.mkdir(EXPERIMENT_FOLDER)
                    os.mkdir(EXPERIMENT_FOLDER+'/w')
                    os.mkdir(EXPERIMENT_FOLDER+'/plots')
                    os.mkdir(EXPERIMENT_FOLDER+'/logs')
                

                    
                
                
                # %% Opt loop
                for iterator in range(ITER_OPS):
                
                    RESNET = False 
                                        
                    if DISCRETE :
                        N_ITERATIONS_TRAIN = int(3e3)
                        if MSGS_ACT == 4:
                            NUM_LAYERS =  8
                        else:
                            NUM_LAYERS =  5 #np.random.randint(3, 15)
                        
                        LAYER_SIZE =  1100 #np.random.randint(100, 500) 
                    else:
                        
                        N_ITERATIONS_TRAIN = int(8e3)
                        # Layer params
                        NUM_LAYERS =  8 #np.random.randint(2, 8)
                        LAYER_SIZE =  300 #np.random.randint(16, 356) 
                    
                    
                    N_LOG_INTERVAL = int(N_ITERATIONS_TRAIN/5)
                        
                    # %%% iteration params
                    t = time.localtime()
                    timestamp = time.strftime('%H:%M:%S', t)
                    
                    # PARAMETERS
                    # DQN and DDPG Params 
                    SEED = [np.random.randint(1, 100) for _ in range(25)]
              
                    BATCH_SIZE = 128 
                    LEARNING_RATE = 1e-3
                    TARGET_MODEL_UPDATE = 1e-3 
                    # Number of steps before the backward method starts the update
                    N_WARMUP = 500 
                    MEMORY_SIZE = int(1e4)
                    MEMORY_WINDOW = 1
                    CLIPNORM =  1e-4 
                
                    
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
                        "SELFISH": SELFISH,
                        "RESNET": RESNET
                        }
                    
                    # Environment and seeds intialization
                    env = AirportEnv(n_agents=N_AGENTS, max_dem=MAX_DEM, msg_act=MSGS_ACT,
                                     rand_graph=RAND_GRAPH, graph=GRAPH, is_discrete = DISCRETE)
                    env.seed(SEED[1])
                    n_actions_per_agent = env.act_per_agent
                    n_obs_per_agent = env.obs_per_agent
                    model_architecture = [LAYER_SIZE for i in range(NUM_LAYERS)]
                
                    # %%% Build agents
                
                    def build_actor(model_architecture_):
                
                        actor = Sequential(name='actor')
                        actor.add(Flatten(input_shape=(MEMORY_WINDOW, n_obs_per_agent)))
                        for layer, u in enumerate(model_architecture_):
                            initializer = tf.keras.initializers.GlorotUniform(seed=SEED[layer])
                            actor.add(Dense(units=u, kernel_initializer=initializer,kernel_regularizer=l2(0.01)))
                            actor.add(Activation('relu'))
                            #actor.add(BatchNormalization())
                
                        actor.add(Dense(n_actions_per_agent, kernel_initializer=initializer))
                        actor.add(Activation('tanh'))
                
                        return actor
                
                    def build_actor2(model_architecture_):
                
                        observation_input = Input(shape=(MEMORY_WINDOW, n_obs_per_agent),
                                                  name='actor_observation_input')
                        if RESNET:
                            model_architecture_ = model_architecture_[:int(len(model_architecture_)/2)]
                            
                        x = Flatten()(observation_input)
                        for layer, u in enumerate(model_architecture_):
                            initializer = tf.keras.initializers.GlorotUniform(seed=SEED[layer])
                            x = Dense(units=u, kernel_initializer=initializer)(x)
                            x = Activation('relu')(x)
                            if RESNET:
                                y = Dense(units=u, kernel_initializer=initializer)(x)
                                y = Activation('relu')(y)
                                x = Add()([x,y]) # ResNet
                            
                            #x = BatchNormalization()(x)
                
                        x = Dense(n_actions_per_agent, kernel_initializer=initializer)(x)
                        x = Activation('tanh')(x)
                        actor = Model(inputs=observation_input, outputs=x, name='actor')
                        return actor
                
                    def build_critic(model_architecture_):
                
                        action_input = Input(shape=(n_actions_per_agent,), name='action_input')
                        observation_input = Input(shape=(MEMORY_WINDOW, n_obs_per_agent),
                                                  name='critic_observation_input')
                
                        if RESNET:
                            model_architecture_ = model_architecture_[:int(len(model_architecture_)/2)]
                
                        flattened_observation = Flatten()(observation_input)
                        
                
                        for lay, u in enumerate(model_architecture_):
                            initializer = tf.keras.initializers.GlorotUniform(seed=SEED[-lay])
                            if lay == 0: 
                                x = Dense(units=u, kernel_initializer=initializer)(flattened_observation)
                            else:         
                                x = Dense(units=u, kernel_initializer=initializer)(x)
                            x = Activation('relu')(x)
                            if RESNET:
                                y = Dense(units=u, kernel_initializer=initializer)(x)
                                y = Activation('relu')(y)
                                x = Add()([x,y]) # ResNet
                                x = BatchNormalization()(x)
                            if lay == 2 :
                                x = Concatenate()([action_input, x])
                                
                
                        x = Dense(1, kernel_initializer=initializer)(x)
                        critic = Model(inputs=[action_input, observation_input],
                                       outputs=x, name='critic')
                
                        return critic, action_input
                    
                    def build_critic2(model_architecture_):
                
                        action_input = Input(shape=(n_actions_per_agent,), name='action_input')
                        observation_input = Input(shape=(MEMORY_WINDOW, n_obs_per_agent),
                                                  name='critic_observation_input')
                
                        if RESNET:
                            model_architecture_ = model_architecture_[:int(len(model_architecture_)/2)]
                
                        flattened_observation = Flatten()(observation_input)
                        x = Concatenate()([action_input, flattened_observation])
                
                        for lay, u in enumerate(model_architecture_):
                            initializer = tf.keras.initializers.GlorotUniform(seed=SEED[-lay])
                            x = Dense(units=u, kernel_initializer=initializer)(x)
                            x = Activation('relu')(x)
                            if RESNET:
                                y = Dense(units=u, kernel_initializer=initializer)(x)
                                y = Activation('relu')(y)
                                x = Add()([x,y]) # ResNet
                                x = BatchNormalization()(x)
                
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
                    
                
                    tf.keras.backend.clear_session()
                    
                    memory = SequentialMemory(limit=MEMORY_SIZE, window_length=MEMORY_WINDOW)
                    
                    if not DISCRETE: 
                        actor = build_actor2(model_architecture)
                    
                        critic, action_input = build_critic2(model_architecture)
                    
                    
                        
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
                        
                        dqn_net = build_dqn_model(model_architecture)
                        
                        policy = BoltzmannQPolicy()
                        agent =  DQNAgent(model=dqn_net, nb_actions=36, memory=memory, nb_steps_warmup=N_WARMUP,
                                          target_model_update=TARGET_MODEL_UPDATE, policy=policy, enable_double_dqn=True)
                
                    agents_list = []
                    opts = []
                    for i in range(N_AGENTS):
                        agents_list.append(agent)
                        opts.append(Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM))
                
                    agent = MultiAgent(agents_list, selfish=SELFISH, n_obs_per_agent = n_obs_per_agent) 
                
                    agent.compile(opts, metrics=['mae'])
                
                    # %%% Train
                
                    # agent.load_weights("./weights/dqn_24-141_agents-3_maxDem-3_selfish-0.591_msg-True_21-07-29_22.31.09_weights")
                    print(f'iterator: {iterator}')
                    train = agent.fit(env, nb_steps=N_ITERATIONS_TRAIN,
                                      log_interval=N_LOG_INTERVAL, visualize=False,
                                      verbose=VERBOSE, nb_max_episode_steps=MAX_STEPS_EPISODE)
                
                    
                
                    # %%% Test
                    
                    env.reset_delays()
                    
                    # Finally, evaluate our algorithm.
                    test = agent.test(env, nb_episodes=N_EPISODES_TEST, visualize=True,
                                      verbose=VERBOSE, nb_max_episode_steps=MAX_STEPS_EPISODE)
                
                    
                    delays = env.get_delays()
                    
                    delays_per_step_agent = tuple(d/(N_EPISODES_TEST*N_AGENTS*MAX_STEPS_EPISODE) for d in delays)
                    
                    env.close()
                
                    mean_q_step, mean_steps, mean_ep_rw = pTT.compute_means(test=test)
                    # %%% Save history
                    t = time.localtime()
                    timestamp = time.strftime('%y-%m-%d_%H.%M.%S', t)
                    
                    
                    if DISCRETE:
                        FILE_NAME = EXPERIMENT_FOLDER + '/w/dqn_agents-'+str(N_AGENTS)+'_sfh-'+str(SELFISH)+'_msg-'+str(MSGS_ACT)+'_'+timestamp+'_weights.h5f'
                    else:
                        FILE_NAME = EXPERIMENT_FOLDER + '/w/ddpg_agents-'+str(N_AGENTS)+'_sfh-'+str(SELFISH)+'_msg-'+str(MSGS_ACT)+'_'+timestamp+'_weights.h5f'
                
                    
                    agent.save_weights(FILE_NAME, overwrite=True)
                    
                    
                    train_dict = {}
                    
                    train_dict['nb_steps'] = [int(i) for i in train.history['nb_steps']]
                    train_dict['episode_rewardtest'] = [float(i) for i in train.history['episode_reward']]
                    train_dict['nb_episode_steps'] = [int(i) for i in train.history['nb_episode_steps']]
                    
                    FILE_NAME_TRAIN = EXPERIMENT_FOLDER+'/logs/train_log_'+timestamp+'.json'
                    with open(FILE_NAME_TRAIN, 'w') as convert_file:
                        convert_file.write(json.dumps(train_dict, indent=3))
                    
                    test_dict = {}
                    
                    test_dict['nb_steps'] = [int(i) for i in test.history['nb_steps']]
                    test_dict['episode_reward'] = [float(i) for i in test.history['episode_reward']]
                
                    
                    FILE_NAME_TEST =  EXPERIMENT_FOLDER+'/logs/test_log_'+timestamp+'.json'
                    with open(FILE_NAME_TEST, 'w') as convert_file:
                        convert_file.write(json.dumps(test_dict, indent=3))
                        
                
                    # %%% Prints
                    exe_params["MEAN_Q_STEP"] = mean_q_step
                    exe_params["MEAN_STEPS"] = mean_steps
                    exe_params["MEAN_EP_RW"] = mean_ep_rw
                
                    rand_param_search.append((exe_params["MEAN_EP_RW"], exe_params, delays_per_step_agent))
                    rand_param_search.sort(reverse=True)
                    # %%% Prints 2
                
                    print('\n***********************************************')
                    print(f'Iteration: {iterator} finished at: {timestamp}')
                    print(f' Top 1 params so far give mean ep. rw = {rand_param_search[0][0]} | Discrete: {rand_param_search[0][1]["DISCRETE"]} | Messaging: {rand_param_search[0][1]["MESSAGES_ACTIVE"]} | Selfish index: {rand_param_search[0][1]["SELFISH"]} | Layers: {rand_param_search[0][1]["NUM_LAYERS"]}| Layer size: {rand_param_search[0][1]["NUM_UNITS_PER_LAYER"]} ')
                    print(f' Top 2 params so far give mean ep. rw = {rand_param_search[1][0]} | Discrete: {rand_param_search[1][1]["DISCRETE"]} | Messaging: {rand_param_search[1][1]["MESSAGES_ACTIVE"]} | Selfish index: {rand_param_search[1][1]["SELFISH"]} | Layers: {rand_param_search[1][1]["NUM_LAYERS"]}| Layer size: {rand_param_search[1][1]["NUM_UNITS_PER_LAYER"]} ')
                    print(f' Top 3 params so far give mean ep. rw = {rand_param_search[2][0]} | Discrete: {rand_param_search[2][1]["DISCRETE"]} | Messaging: {rand_param_search[2][1]["MESSAGES_ACTIVE"]} | Selfish index: {rand_param_search[2][1]["SELFISH"]} | Layers: {rand_param_search[2][1]["NUM_LAYERS"]}| Layer size: {rand_param_search[2][1]["NUM_UNITS_PER_LAYER"]} ')
                    print(f' Top 4 params so far give mean ep. rw = {rand_param_search[3][0]} | Discrete: {rand_param_search[3][1]["DISCRETE"]} | Messaging: {rand_param_search[3][1]["MESSAGES_ACTIVE"]} | Selfish index: {rand_param_search[3][1]["SELFISH"]} | Layers: {rand_param_search[3][1]["NUM_LAYERS"]}| Layer size: {rand_param_search[3][1]["NUM_UNITS_PER_LAYER"]} ')
                    print(f' Top 5 params so far give mean ep. rw = {rand_param_search[4][0]} | Discrete: {rand_param_search[4][1]["DISCRETE"]} | Messaging: {rand_param_search[4][1]["MESSAGES_ACTIVE"]} | Selfish index: {rand_param_search[4][1]["SELFISH"]} | Layers: {rand_param_search[4][1]["NUM_LAYERS"]}| Layer size: {rand_param_search[4][1]["NUM_UNITS_PER_LAYER"]} ')
                    print('***********************************************\n')
                
                
                
                    # %%Write file
                    FILE_NAME_PARAMS = EXPERIMENT_FOLDER+'/'+str(rand_param_search[0][1]["MEAN_EP_RW"])+'_'+timestamp+'.json'
                    # FILE_NAME_PARAMS = 'ops_exe_param_noche_25-07-21_v2.json'
                    if iterator == ITER_OPS-1:
                        
                        with open(FILE_NAME_PARAMS, 'w') as convert_file:
                                convert_file.write(json.dumps(rand_param_search, indent=3))
                
                
                # %% Test for log
                
                    test_env = AirportEnv(n_agents=N_AGENTS, max_dem = MAX_DEM, msg_act = MSGS_ACT, rand_graph = RAND_GRAPH, log_act = True, graph = GRAPH, is_discrete=DISCRETE)
                    
                    test_env.seed(SEED)
                    
                    
                    tst = agent.test(test_env, nb_episodes=1, visualize=True,
                                      verbose=VERBOSE, nb_max_episode_steps=20)
                
                
                        
                    env_log = test_env.return_env_log()
                    
                    log = logTool.process_env_log(env_log)
                    
                # %%
                    for a in range(N_AGENTS):
                        pTT.plot_train_test(train = train, test = test, log = log, n=int(1e6), exe_params = exe_params, t= timestamp, agent = a, folder = EXPERIMENT_FOLDER)
                    
                    
                # %%
                    
                    FILE_NAME_LOG = EXPERIMENT_FOLDER+'/logs/demo_'+timestamp+'.json'
                    
                    with open(FILE_NAME_LOG, 'w') as convert_file:
                        convert_file.write(json.dumps(log, indent=3))
                
                    del agent
                # %%
                # FILE_NAME = './experiments/21-07-29 selfish and msg/params/opt_exe_param_1556.416663_21-07-29_04.21.55'
                # f = open(FILE_NAME+'.json',)
                # rand_param_search = json.load(f)
                # agent.load_weights("./weights/dqn_24-141_agents-3_maxDem-3_selfish-0.591_msg-True_21-07-29_22.31.09_weights.h5f")
                
                # pTT.plot_rand_params(rand_param_search)
            
