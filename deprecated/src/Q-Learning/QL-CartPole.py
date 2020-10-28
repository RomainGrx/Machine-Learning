# -*- coding: utf-8 -*-

import os
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# GLOBAL VARIABLES
env = gym.make('CartPole-v0')
discount_factor = 0.95 # discount factor
render_frequency = 1000
epochs = 25_000

discrete_observation_shape = (1, 1, 25, 25)

epsilon_high = 0.95
epsilon_low = 0.1
epsilon_decay = epochs//4

learning_rate_high = 0.5
learning_rate_low = 0.1
learning_rate_decay = epochs//10

ENV_MINMAX = 5

upper_bounds = np.array([env.observation_space.high[0], ENV_MINMAX, env.observation_space.high[2], ENV_MINMAX])
lower_bounds = np.array([env.observation_space.low[0], -ENV_MINMAX, env.observation_space.low[2], -ENV_MINMAX])

def get_finite_state(state):
    discrete_state = ((state - lower_bounds)/ (upper_bounds - lower_bounds))* discrete_observation_shape
    return tuple(discrete_state.astype(int))

if not os.path.exists('save_numpy'):
    os.makedirs('save_numpy')

NAME = f'QL-CartPole_{discrete_observation_shape}_MIN_MAX_{ENV_MINMAX}_LRMIN={learning_rate_low}_LRMAX={learning_rate_high}_LRDECAY={learning_rate_decay}_EPMIN={epsilon_low}_EPMAX={epsilon_high}_EPDECAY={epsilon_decay}'
WRITER = tf.summary.create_file_writer(f'tensorboard/{NAME}')

epsilon = epsilon_high
learning_rate = learning_rate_high

qtable = np.random.uniform(low=-2, high=0, size=(discrete_observation_shape + (env.action_space.n,)))

reward_list = []
loss_list = []

MIN_1, MAX_1 = 0, 0
MIN_3, MAX_3 = 0, 0

for episode in range(epochs):
    done = False
    reward_values = []
    loss_values = []
    finite_state = get_finite_state(env.reset())
    step = 0
    while not done:
        if episode % render_frequency == 0: env.render()
        if random.random() < epsilon:
            action = random.randint(0, env.action_space.n-1)
        else:
            action = np.argmax(qtable[finite_state])
        next_state, reward, done, info = env.step(action)

        next_finite_state = get_finite_state(next_state)
        target_q = (reward + discount_factor * np.max(qtable[next_finite_state + (action,)])) if not done else reward

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

    print(f'episode {episode+1}/{epochs} : reward {reward_list[-1]} : loss {loss_list[-1]:.2f}: step {step} : epsilon {epsilon:.3f} : lr {learning_rate:.3f}')

    epsilon = epsilon_low + (epsilon_high - epsilon_low)*np.exp(-episode/epsilon_decay)
    learning_rate = learning_rate_low + (learning_rate_high - learning_rate_low)*np.exp(-episode/learning_rate_decay)

np.save(f'save_numpy/{NAME}', qtable)
env.close()
