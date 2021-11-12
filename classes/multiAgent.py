import numpy as np
import warnings
from copy import deepcopy

from rl.core import Agent
from keras.layers import Concatenate
from os.path import splitext

from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)



from tensorflow.keras.callbacks import History

class MultiAgent(Agent):
    def __init__(self, agents, n_obs_per_agent, selfish = 0):
        self.agents = agents
        self.cur_agent = -1
        self.n_agents = len(agents)

        self.compiled = False
        self.m_names = []
        self._training = False
        self._step = 0       
        self.obs_per_agent = n_obs_per_agent
        self.selfish_index = selfish
        
        
        super(MultiAgent, self).__init__()

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self,t):
        self._training = t
        for agent in self.agents:
            agent.training = t

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self,s):
        #print "setting step %i" % s
        self._step = s
        for agent in self.agents:
            agent.step = s
    
    def reset_states(self):
        self.cur_agent = 1
        for agent in self.agents:
            agent.reset_states()

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.
        # Argument
            observation (object): The current observation from the environment.
        # Returns
            The next action to be executed in the environment.
        """
        
        
        observation = np.array(observation)
        obs_mtx = observation.reshape((self.n_agents, self.obs_per_agent))
        
        actions = []
        for i in range(self.n_agents):
            obs_agent = obs_mtx[i]
            obs_agent = obs_agent.T
            agent_action = self.agents[i].forward(obs_agent)
            if isinstance(agent_action, np.int32) or isinstance(agent_action, np.int64):
                actions.append(agent_action)
            else:
                actions.extend(agent_action)

        return actions

    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.
        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        # Returns
            List of metrics values
        """
        global_rw = reward[0]
        list_metrics = []
        for i in range(self.n_agents):
            
            local_rw = reward[1][i]
            
            mean_global_reward = global_rw/self.n_agents
            
            weighted_reward = local_rw * self.selfish_index + mean_global_reward * (1-self.selfish_index)
            
            list_metrics = self.agents[i].backward(weighted_reward,terminal)[:len(self.m_names)]
        
        return list_metrics

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.
        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        for i,agent in enumerate(self.agents):
            if not agent.compiled:
                agent.compile(optimizer[i],metrics)

        # Just truncate the list of metrics if one has more (assume prefixes match)
        if len(self.agents) > 1:
            if len(self.agents[0].metrics_names)<=len(self.agents[1].metrics_names):
                self.m_names = self.agents[0].metrics_names
            else:
                self.m_names = self.agents[1].metrics_names
        else:
            self.m_names = self.agents[0].metrics_names
            #print(self.agents[0].metrics_names)

        self.compiled = True

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.
        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        fbase, fext = splitext(filepath)
        for i, agent in enumerate(self.agents):
            agent.load_weights('%s%i%s' % (fbase,i,fext))

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.
        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        fbase, fext = splitext(filepath)
        for i, agent in enumerate(self.agents):
            agent.save_weights('%s%i%s' % (fbase,i,fext), overwrite)

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).
        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.
        # Returns
            A list of the model's layers
        """
        return [ layer for agent in self.agents
                    for layer in agent.layers() ]

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        # Returns
            A list of metric's names (string)
        """
        # Assumes all agents share metric names
        return self.m_names

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        for agent in self.agents:
            agent._on_train_begin()

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        for agent in self.agents:
            agent._on_train_end()

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        for agent in self.agents:
            agent._on_test_begin()

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        for agent in self.agents:
            agent._on_test_end()
            
    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError(f'action_repetition must be >= 1, is {action_repetition}')

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn(f'Env ended before {nb_random_start_steps} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.')
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    
                    # Modifications here!
                    
                    reward = r
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                
                # Modifications here!
                
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward[0]

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward[0],
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    place_holder_rw = (0, [0]*self.n_agents)
                    self.backward(place_holder_rw, terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
                    
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history
    
    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
         nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.
    
        #  Arguments
        env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
        nb_episodes (integer): Number of episodes to perform.
        action_repetition (integer): Number of times the agent repeats the same action without
            observing the environment again. Setting this to a value > 1 can be useful
            if a single action only has a very small effect on the environment.
        callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
            List of callbacks to apply during training. See [callbacks](/callbacks) for details.
        verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
        visualize (boolean): If `True`, the environment is visualized during training. However,
            this is likely going to slow down training significantly and is thus intended to be
            a debugging instrument.
        nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
            of each episode using `start_step_policy`. Notice that this is an upper limit since
            the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
            at the beginning of each episode.
        start_step_policy (`lambda observation: action`): The policy
            to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
        log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
        nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
            automatically resetting the environment. Set to `None` if each episode should run
            (potentially indefinitely) until the environment signals a terminal state.
    
        # Returns
        A `keras.callbacks.History` instance that recorded the entire training process.
        """
    
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError(f'action_repetition must be >= 1, is {action_repetition}')
    
        self.training = False
        self.step = 0
    
        callbacks = [] if not callbacks else callbacks[:]
    
        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
    
        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0
    
            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None
    
            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn(f'Env ended before {nb_random_start_steps} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.')
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    break
    
            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)
    
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward = r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward[0]
    
                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1
    
            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            place_holder_rw = (0, [0]*self.n_agents)
            self.backward(place_holder_rw, terminal=False)
    
            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()
    
        return history