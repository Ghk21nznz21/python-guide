import random
from collections import deque
from copy import copy
from dataclasses import KW_ONLY, dataclass

import gym
import numpy as np
from attr import field
from IPython.display import clear_output
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam


@dataclass(KW_ONLY=True)
class Interaction:
    env: object  # TDP, STATES, REWARDS 
    agent: object 
    alpha: float = field(default=0.9)
    gamma: float = field(default=0.9) #learning rate 
    #gamma_decay: float = field(default=0.9) cva
    epsilon: float = field(default=1) 
    epsilon_decay: float = field(default=0.01) 
    episodes: float = field(default=0.9)
    max_iter: any = field(default=None)
    batch_size: int = field(default=30)


    def q_learning(self):
        all_rewards=[] 
        self.env.render() 
        for episode in self.episodes:
            self.env.reset()
            ep_reward= 0
            done=False
            for _ in self.max_iter:
                #agent - action - env 
                action =  self.agent.take_action(self.env)
                next_state, reward, done, info = self.env.step(action) 

                #update image
                self.env.render()

                #Calculate and Update Q_matrix
                self.agent.update_Q(next_state, reward, self.alpha, self.gamma)
        
                #Update Enviroment
                ep_reward += reward 
                state = next_state

                if done:
                    break 
            all_rewards.append(ep_reward)
            self.epsilon_decay()
            self.env.render() 
        return all_rewards 
    
    def deep_q(self):
        all_rewards=[] 
        self.env.render() 
        for episode in self.episodes:
            self.env.reset()
            ep_reward= 0
            done=False
            for _ in self.max_iter:
                #agent - action - env 
                action =  self.agent.take_action(self.env)
                next_state, reward, done, info = self.env.step(action) 

                #update image
                self.env.render()

                #Calculate and Update Q_matrix
                self.agent.update_Q(next_state, reward, self.alpha, self.gamma)
        
                #Update Enviroment
                ep_reward += reward 
                state = next_state

                if done:
                    self.agent.alighn_target_model()
                if len(self.agent.expirience_replay) > self.batch_size:
                    self.agent.retrain(self.batch_size)
            all_rewards.append(ep_reward)
            self.epsilon_decay()
            self.env.render() 
        return all_rewards 
    



    def decay_epsilon(self):
        self.epsilon -= self.epsilon_decay
        #ex: np.exp(-decay_rate*episode)


@dataclass
class Enviroment: #home made 
    states: list #matrix
    rewards: list #matrix
    tdp: list #matrix

    def step(self):
        pass #return next_state, reward, done, info


@dataclass
class Agent:
    _action_size: int 
    _state_size: int 
    _Q: list = field(init=False, default= np.zeros(len(_state_size), len(_action_size)) )

    def take_action(self, env: object, table: list, state: int, epsilon=0.9):
        if np.random.uniform(0, 1) <= epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self._Q[state])

    def update_Q(self, state: any, action: any, next_state: any, reward: float, 
            alpha: float, gamma: float ):
        q_value = self._Q[state, action]
        max_value = np.max(self._Q[next_state])
        new_q_value = (1 - alpha)*q_value + alpha*(reward + gamma*max_value)
        self._Q[state, action] = new_q_value


@dataclass
class DeepAgent:
    _action_size: int 
    _state_size: int 
    _optimizer: str  
    experience_replay: object = field(init=False, default= deque(maxlen=10))
    behaviour_policy: object  = field(init=False) 
    target_policy: object = field(init=False)

    def __pos_init__(self):
        self.behaviour_policy = self._build_compile_model()
        self.target_network = self.behaviour_policy.copy()

    def take_action(self, env: object, updater: object, state: any, epsilon=0.9):
        if random.uniform(0, 1) <= epsilon:
            return  env.action_space.sample()
        else:
            q_values = self.behaviour_policy.predict(state)
            return np.argmax(q_values[0])

    def _build_compile_model(self):
        #init = tf.keras.initializers.HeUniform()
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
            # Represent 500 possible states by 10 values
        model.add(Reshape((10,))) #prepares data for Dense Layers
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def store(self, state: any, action: any, reward: float, next_state: any, done: bool):
        self.experience_replay.append((state, action, reward, next_state, done))

    def update(self, batch_size: int):
        '''
        The agent will play the game in a first phase,
        and then it will be trained on that experience.
        This corresponds to the train/update part 
        '''
        minibatch = random.sample(self.experience_replay, batch_size)
        for state, action, reward, next_state, done in minibatch:
            q = self.behaviour_policy.predict(state) #.........
            if done:
                q[0][action] = reward
            else:
                next_q = self.target_network.predict(next_state)
                q[0][action] = reward + self.gamma * np.amax(next_q)
            self.behaviour_policy.fit(state, q, epochs=1, verbose=0)
