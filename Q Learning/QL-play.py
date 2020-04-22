#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : Monday, 20 April 2020
"""

import numpy as np
import gym

# Global variables

# GAME = 'MountainCar-v0'
GAME = 'CartPole-v0'
EPOCHS = 10

env = gym.make(GAME)

if GAME == 'MountainCar-v0':
    Q_TABLE_NAME = 'QL-MountainCar_(25, 25)_LRMIN=0.05_LRMAX=0.3_LRDECAY=2500_EPMIN=0.1_EPMAX=0.95_EPDECAY=6250.npy'
    discrete_observation_shape = (25,25)
    upper_bounds = env.observation_space.high
    lower_bounds = env.observation_space.low
elif GAME == 'CartPole-v0':
    Q_TABLE_NAME = 'QL-CartPole_(1, 1, 20, 20)_MIN_MAX_5.npy'
    discrete_observation_shape = (1,1,20,20)
    ENV_MINMAX = 5
    upper_bounds = np.array([env.observation_space.high[0], ENV_MINMAX, env.observation_space.high[2], ENV_MINMAX])
    lower_bounds = np.array([env.observation_space.low[0], -ENV_MINMAX, env.observation_space.low[2], -ENV_MINMAX])

# Program

def get_finite_state(state):
    discrete_state = ((state - lower_bounds)/ (upper_bounds - lower_bounds))* discrete_observation_shape
    return tuple(discrete_state.astype(int))

qtable = np.load(f'save_numpy/{Q_TABLE_NAME}')

for episode in range(EPOCHS):
    state = env.reset()
    done = False
    while not done:
        env.render()
        finite_state = get_finite_state(state)
        action = np.argmax(qtable[finite_state])
        state, _, done, _ = env.step(action)
        finite_state = get_finite_state(state)
env.close()
