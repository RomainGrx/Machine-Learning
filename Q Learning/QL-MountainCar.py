#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : Friday, 17 April 2020
"""

import os
import gym
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Global variables
env = gym.make('MountainCar-v0')

discount_factor = 0.95 # discount factor
render_frequency = 1000
epochs = 25_000

discrete_observation_shape = (25, 25)

epsilon_high = 0.95
epsilon_low = 0.1
epsilon_decay = epochs//4

learning_rate_high = 0.1
learning_rate_low = 0.1
learning_rate_decay = epochs//10

# Training

def get_finite_state(state):
    discrete_state = ((state - env.observation_space.low)/ (env.observation_space.high - env.observation_space.low))* discrete_observation_shape
    return tuple(discrete_state.astype(int))

if not os.path.exists('save_numpy'):
    os.makedirs('save_numpy')

NAME = f'QL-MountainCar_{discrete_observation_shape}_LRMIN={learning_rate_low}_LRMAX={learning_rate_high}_LRDECAY={learning_rate_decay}_EPMIN={epsilon_low}_EPMAX={epsilon_high}_EPDECAY={epsilon_decay}'
WRITER = tf.summary.create_file_writer(f'tensorboard/{NAME}')

epsilon = epsilon_high
learning_rate = learning_rate_high

qtable = np.random.uniform(low=-2, high=0, size=(discrete_observation_shape + (env.action_space.n,)))
# qtable = np.load('./save_numpy/baseline_without_epsilon_(20, 20, 3).npy') ; epsilon = 0; epsilon_low=0; epsilon_high=0

reward_list = []
loss_list = []

for episode in range(epochs):
    done = False
    reward_values = []
    loss_values = []
    finite_state = get_finite_state(env.reset())
    step = 0
    REACH = False
    while not done:
        if episode % render_frequency == 0: env.render()
        if random.random() < epsilon:
            action = random.randint(0, env.action_space.n-1)
        else:
            action = np.argmax(qtable[finite_state])
        next_state, reward, done, info = env.step(action)

        if next_state[0] > env.goal_position:
            REACH = True
            qtable[finite_state + (action,)] = 0
            loss = 0
        else:
            next_finite_state = get_finite_state(next_state)
            target_q = (reward + discount_factor * np.max(qtable[next_finite_state + (action,)]))
            loss = abs(qtable[finite_state + (action,)] - target_q)
            qtable[finite_state + (action,)] = (1 - learning_rate)*qtable[finite_state + (action,)] + learning_rate * target_q

        loss_values += [loss]
        reward_values += [reward]

        step += 1
        finite_state = next_finite_state

    reward_list += [np.sum(reward_values)]
    loss_list += [np.mean(loss_values)]

    with WRITER.as_default():
        tf.summary.scalar("loss", loss_list[-1], step=episode)
        tf.summary.scalar("reward", reward_list[-1], step=episode)
        tf.summary.scalar("reach_goal", int(REACH), step=episode)
        tf.summary.scalar("learning_rate", learning_rate, step=episode)
        tf.summary.scalar("epsilon", epsilon, step=episode)

    print(f'episode {episode+1}/{epochs} : reward {reward_list[-1]} : loss {loss_list[-1]:.2f}: step {step} : epsilon {epsilon:.3f} : lr {learning_rate:.3f} : reach goal {REACH}')

    epsilon = epsilon_low + (epsilon_high - epsilon_low)*np.exp(-episode/epsilon_decay)
    learning_rate = learning_rate_low + (learning_rate_high - learning_rate_low)*np.exp(-episode/learning_rate_decay)

np.save(f'save_numpy/{NAME}', qtable)
env.close()

def plot_array(array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    m, n, l = array.shape
    low = env.observation_space.low
    high = env.observation_space.high
    x,y,z = np.meshgrid(np.linspace(low[0], high[0], m), np.linspace(low[1], high[1], n), np.arange(l))
    sca = ax.scatter(x,y,z, c=array)
    ax.set_zticks([0,1,2]); ax.set_zticklabels(['Left', 'No push', 'Right'])
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Action')
    cbar=plt.colorbar(sca)
    plt.show()

def plot_max_array(array):
    fig = plt.figure()
    plt.title('Representation by action')
    m, n, l = array.shape
    low = env.observation_space.low
    high = env.observation_space.high
    argmax = np.argmax(array, axis=2)
    ret = plt.imshow(argmax)
    plt.xticks(np.arange(m), list(map(lambda x:'%.2f'%x, np.linspace(low[0], high[0], m))))
    plt.yticks(np.arange(n), list(map(lambda x:'%.2f'%x, np.linspace(low[1], high[1], n))))
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    cbar=plt.colorbar(ret)
    cbar.set_ticks([0,1,2])
    cbar.set_ticklabels(['Left', 'No push', 'Right'])
    plt.show()

plot_max_array(qtable)
# plot_array(qtable)
