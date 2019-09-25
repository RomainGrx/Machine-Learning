#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:16:55 2019

@author: romaingraux
"""



# import tflearn as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

def adjust(state):
    return np.round(state * np.array([10, 100])).astype(int)

env = gym.make('MountainCar-v0')
episodes = 1000
LR = 0.2
gamma = 0.9
min_eps = 0
epsilon = 0.5
reward_list = []
ave_reward_list = []
max_reward_list = []
reduction = epsilon / episodes

try:
    states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
    states = np.round(states).astype(int) + 1
    Q = np.zeros((states[0], states[1], env.action_space.n))
    for e in range(episodes):
            rewardTot, reward = 0, 0
            state = env.reset()
            state = adjust(state)
            done = False
            while done != True:
                if (e+1) % 500 == 0 : env.render()
                if np.random.random() < 1 - epsilon :
                    action = np.argmax(Q[state[0], state[1]])
                else :
                    action = np.random.randint(0, env.action_space.n)
                state2, reward, done, info = env.step(action)
                state2 = adjust(state2)
                delta = LR*(reward + gamma*np.max(Q[state2[0], state2[1]]) - Q[state[0], state[1],action])
                Q[state[0], state[1], action] += delta
                state = state2
                rewardTot += reward
                epsilon -= reduction
            reward_list.append(rewardTot)
            if (e+1) % 100 == 0:
                print('Episode {} Average Reward: {} Maximum Reward: {}'.format(e+1, np.mean(reward_list), np.max(reward_list)))
                ave_reward_list.append(np.mean(reward_list))
                max_reward_list.append(np.max(reward_list))
                reward_list = []

    plt.plot(100*(np.arange(len(ave_reward_list)) + 1), ave_reward_list, 'y-')
    plt.plot(100*(np.arange(len(ave_reward_list)) + 1), max_reward_list, 'r-')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.show()

finally:
    env.close()





