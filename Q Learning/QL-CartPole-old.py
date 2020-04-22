#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:24:31 2019

@author: romaingraux
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import math

class CartPole():

    def __init__(self, buckets=(1, 1, 6, 12,), n_episodes = 201, LR = 1.0, epsilon = 1.0, gamma = 1.0, min_LR = 0.1, min_epsilon = 0.1, show = False):
        self.buckets = buckets
        self.n_episodes = n_episodes
        self.LR = LR
        self.epsilon = epsilon
        self.gamma = gamma
        self.min_LR = min_LR
        self.min_epsilon = min_epsilon
        self.env = gym.make('CartPole-v0')
        self.Q = np.zeros(buckets + (self.env.action_space.n,), dtype = np.float64)
        self.min_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        self.max_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        self.time = 0
        self.current_episode = 0
        self.show = show
        self.ave_reward_list = []
        self.rewardTot_list = []

    def adjust(self, state):
        ratio = [(state[i] + abs(self.min_bounds[i])) / (self.max_bounds[i] - self.min_bounds[i]) for i in range(len(state))]
        new_state = [int(round((self.buckets[i] - 1) * ratio[i])) for i in range(len(state))]
        new_state = [min(self.buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
        return tuple(new_state)

    def iter(self):
        self.epsilon = max(self.min_epsilon, min(1, 1.0 - math.log10((self.current_episode + 1) / 25)))
        self.LR = max(self.min_LR, min(1.0, 1.0 - math.log10((self.current_episode + 1) / 25)))

    def action(self, state):
        return self.env.action_space.sample() if (np.random.random() <= self.epsilon) else np.argmax(self.Q[state])

    def upgradeQ(self, old_state, new_state, action, reward):
        self.Q[old_state][action] += self.LR * (reward + self.gamma * np.max(self.Q[new_state]) - self.Q[old_state][action])

    def plot(self):
        plt.plot(100*(np.arange(len(self.ave_reward_list)) + 1), self.ave_reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episodes')
        plt.show()

    def run(self):
        for self.current_episode in range(self.n_episodes):
            state=  self.env.reset()
            state = self.adjust(state)
            self.time, reward, rewardTot, done = 0, 0, 0, False
            self.iter()
            while not done:
                if (self.current_episode >= ((self.n_episodes - 5))):
                    self.env.render()
                self.time += 1
                if self.show : self.env.render()
                action = self.action(state)
                new_state, reward, done, info = self.env.step(action)
                new_state = self.adjust(new_state)
                self.upgradeQ(state, new_state, action, reward)
                state = new_state
                rewardTot += reward
            self.rewardTot_list.append(rewardTot)
            if (self.current_episode + 1) % 100 == 0:
                self.ave_reward_list.append(np.mean(self.rewardTot_list))
                self.rewardTot_list = []
                print('Episode {} Average Score {} TIME {}'.format(self.current_episode+1, self.ave_reward_list[-1], self.time))
        self.plot()
        self.env.close()

class MountainCar():
    def __init__(self, buckets=(20, 20,), n_episodes = 1500, LR = 1.0, epsilon = 1.0, gamma = 1.0,min_LR = 0.1, min_epsilon = 0.1, show = False):
        self.buckets = buckets
        self.n_episodes = n_episodes
        self.LR = LR
        self.epsilon = epsilon
        self.gamma = gamma
        self.min_LR = min_LR
        self.min_epsilon = min_epsilon
        self.env = gym.make('MountainCar-v0')
        self.Q = np.zeros(buckets + (self.env.action_space.n,), dtype = np.float64)
        self.min_bounds = self.env.observation_space.low
        self.max_bounds = self.env.observation_space.high
        self.time = 0
        self.current_episode = 0
        self.show = show
        self.ave_reward_list = []
        self.rewardTot_list = []
        self.ave_time_list = []
        self.timeTot_list = []


    def adjust(self, state):
        ratio = [(state[i] + abs(self.min_bounds[i])) / (self.max_bounds[i] - self.min_bounds[i]) for i in range(len(state))]
        new_state = [int(round((self.buckets[i] - 1) * ratio[i])) for i in range(len(state))]
        new_state = [min(self.buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
        return tuple(new_state)

    def iter(self):
        self.epsilon = max(self.min_epsilon, min(1, 1.0 - math.log10((self.current_episode + 1) / 25)))
        self.LR = max(self.min_LR, min(1.0, 1.0 - math.log10((self.current_episode + 1) / 25)))

    def action(self, state):
        return self.env.action_space.sample() if (np.random.random() <= self.epsilon) else np.argmax(self.Q[state])

    def upgradeQ(self, old_state, new_state, action, reward):
        self.Q[old_state][action] += self.LR * (reward + self.gamma * np.max(self.Q[new_state]) - self.Q[old_state][action])

    def plot(self):
        plt.plot(100*(np.arange(len(self.ave_reward_list)) + 1), self.ave_reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episodes')
        plt.show()

    def run(self):
        for self.current_episode in range(self.n_episodes):
            state=  self.env.reset()
            state = self.adjust(state)
            self.time, reward, rewardTot, done = 0, 0, 0, False
            self.iter()
            while not done:
                if (self.current_episode >= ((self.n_episodes - 5))):
                    self.env.render()
                self.time += 1
                if self.show : self.env.render()
                action = self.action(state)
                new_state, reward, done, info = self.env.step(action)
                new_state = self.adjust(new_state)
                self.upgradeQ(state, new_state, action, reward)
                state = new_state
                rewardTot += reward
            self.rewardTot_list.append(rewardTot)
            self.timeTot_list.append(self.time)
            if (self.current_episode + 1) % 100 == 0:
                self.ave_reward_list.append(np.mean(self.rewardTot_list))
                self.ave_time_list.append(np.mean(self.timeTot_list))
                self.rewardTot_list = []
                self.timeTot_list
                print('Episode {} Average Score {} TIME {}'.format(self.current_episode+1, self.ave_reward_list[-1], self.ave_time_list[-1]))
        self.plot()
        self.env.close()

MountainCar = MountainCar()
MountainCar.run()
