#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:58:12 2019

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
sys.path.insert(0, '/Users/romaingraux/Documents/Python/Machine-Learning/res/prog')
import useful as u
modelpath = "../../res/models/DQN-CartPole.h5"


class network():
    def __init__(self, env, episodes = 100, batch_size = 32, gamma = 0.95, epsilon = 0.995):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.model = self.build()
        self.memory = deque(maxlen=2000)
        self.gamma = gamma 
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.learning_rate = 0.001
        self.batch_size = batch_size
        self.episodes = episodes
        self.env = env
        self.current_episode = 0


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
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.995)
#        self.epsilon = max(self.min_epsilon, min(1, - 0.7 * math.log10((self.current_episode + 1) / self.episodes)))

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
        state_size = self.state_size
        agent = self
        agent.model = u.loadmodel(modelpath, model = self.model)
        for self.current_episode in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, state_size])
            done = False
            time = 0
            while not done:
                time += 1
                if show : env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                reward = reward if not done else -10
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, epsilon {}"
                          .format(self.current_episode, self.episodes, time, self.epsilon))
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
                output = model.predict(state)
                print(output.shape)
                action = np.argmax(output)
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size])
                if done:
                    print("episode: {}/{}, score: {}"
                          .format(e, episodes, time))
                    break
        env.close()

env = gym.make("CartPole-v0")
network = network(env, episodes = 150, epsilon = 0.995)

# Train the model
network.run(show = True)

# Play with the model trained
# network.play(episodes = 1)











