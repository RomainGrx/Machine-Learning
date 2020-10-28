#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 23:17:55 2019

@author: romaingraux
"""


# import tflearn
import gym
import numpy as np
# import time
import tensorflow.keras as keras
from collections import deque
import random
import sys
import math
sys.path.insert(0, '../../res/prog')
import useful as u
modelpath = "../../res/models/DQN-MountainCar.h5"


class network():
    def __init__(self, env, episodes = 100, batch_size = 32, gamma = 0.95, epsilon = 1.0, epsilon_decay = 0.995):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        self.learning_rate = 0.001
        self.batch_size = batch_size
        self.episodes = episodes
        self.env = env
        self.current_episode = 0
        self.model = self.build()
        self.targetmodel = self.build()
        self.targetmodel.set_weights(self.model.get_weights())


    def build(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return int(random.randrange(self.action_size))
        return np.argmax(self.model.predict(state))

    def eps_adjust(self):
        self.epsilon = max(self.min_epsilon, 1 - self.current_episode / self.episodes)
#        self.epsilon = - 0.0016 * self.current_episode + 0.4
#        self.epsilon = max(self.min_epsilon, min(1, - 0.7 * math.log10((self.current_episode + 1) / self.episodes)))

    def replays(self):
        batch_size = self.batch_size
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.targetmodel.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        self.eps_adjust()
        self.targetmodel.set_weights(self.model.get_weights())

    def replay(self):
        batch_size = self.batch_size
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        self.eps_adjust()

    def run(self, show = False):
        center = -0.55
        state_size = self.state_size
        agent = self
        agent.model = u.loadmodel(modelpath, model = self.model)
        for self.current_episode in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, state_size])
            done = False
            time = 0
            totReward = 200
            while not done:
                time += 1
                if show : env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                cur_x = next_state[0][0] - center
#                reward = -1 if not (done and time !=200) else +10
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                totReward += reward
                if done:
                    print("episode: {}/{}, score: {}, epsilon {}, time {}"
                          .format(self.current_episode, self.episodes, totReward, self.epsilon, time))
                    break
                agent.replay()
        env.close()
        u.modelsave(modelpath, agent.model)
    def play(self, episodes = 5):
        model = u.loadmodel(modelpath, model = None)
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            time = 0
            while not done:
                time += 1
                self.env.render()
                action = np.argmax(model.predict(state))
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size])
                if done:
                    print("episode: {}/{}, score: {}"
                          .format(e, episodes, time))
                    break
        env.close()

env = gym.make("MountainCar-v0")
network = network(env, episodes = 10, epsilon = 1.0, epsilon_decay = 0.9995)

#----------------------------------------------------------------

# Train the model
# network.run(show = True)

# Play with the model trained
network.play(episodes = 10)

#----------------------------------------------------------------
