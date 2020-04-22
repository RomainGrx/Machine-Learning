#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : Tuesday, 21 April 2020
"""

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import matplotlib
print("Gym:", gym.__version__)
print("Tensorflow:", tf.__version__)
print("Matplotlib:", matplotlib.__version__)

# Memory class

Memory_State = namedtuple('Memory_State', ['state', 'action', 'reward', 'next_state', 'done'])
class PrioritizedMemory():
    def __init__(self, size):
        self.memory = deque(maxlen=size)
        self.priorities = deque(maxlen=size)

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        probabilities = scaled_priorities / sum(scaled_priorities)
        return probabilities

    def get_importances(self, probabilities):
        importances = 1/len(self.memory) * 1/probabilities
        importances_normalized = importances / max(importances)
        return importances_normalized

    def push(self, *memory_state):
        self.memory.append(Memory_State(*memory_state))
        self.priorities.append(max(self.priorities, default=1))

    def sample(self, size, priority_scale=1.0):
        size = min(len(self), size)
        probabilities = self.get_probabilities(priority_scale)
        indices = random.choices(np.arange(len(self)), k=size, weights=probabilities)

        sample_memory = np.array(self.memory)[indices]
        importances = self.get_importances(probabilities[indices])
        batch = Memory_State(*zip(*sample_memory))
        return batch, importances, indices

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def __len__(self):
        return len(self.memory)

# Network class

class Network():
    def __init__(self, input_shape, output_n):
        self.input_shape = input_shape
        self.output_n = output_n
        self.policy_network = self.get_model()
        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.optimizer = tf.keras.optimizers.Adam(lr=1E-3)
        self.gamma = .97
        self.tau = .01

    def get_model(self):
        model = tf.keras.Sequential([
                                     tf.keras.layers.Dense(100, input_shape=self.input_shape, activation=tf.nn.relu),
                                     tf.keras.layers.Dense(self.output_n, activation=None)

        ])
        return model

    def act(self, state):
        q_values = self.policy_network(np.expand_dims(state, 0))
        action = np.argmax(q_values)
        return action

    def train(self, batch, importances):

        state_batch = tf.convert_to_tensor(batch.state)
        action_batch = tf.convert_to_tensor(batch.action)
        reward_batch = tf.convert_to_tensor(batch.reward)
        next_state_batch = tf.convert_to_tensor(batch.next_state)
        done_batch = tf.convert_to_tensor(batch.done)

        variables = self.policy_network.trainable_variables

        next_action = tf.math.argmax(self.policy_network(next_state_batch), axis=1)
        q_next_states = self.target_network(next_state_batch)
        next_action_one_hot = tf.one_hot(next_action, depth=self.output_n)
        q_target_action = tf.math.reduce_sum(tf.multiply(q_next_states, next_action_one_hot), axis=1)
        q_target = tf.where(done_batch, reward_batch, reward_batch + self.gamma * q_target_action)

        with tf.GradientTape() as tape:
            tape.watch(variables)

            q_states = self.policy_network(state_batch)

            action_one_hot = tf.one_hot(action_batch, depth=self.output_n)
            q_values_actions = tf.math.reduce_sum(tf.multiply(q_states, action_one_hot) , axis=1)

            errors = q_values_actions - q_target

            loss = tf.reduce_mean(tf.multiply(tf.square(errors), importances))


        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        for t, p in zip(self.target_network.trainable_variables, self.policy_network.trainable_variables):
            t.assign((1-self.tau)*t + self.tau*p)

        return loss, errors

# Agent class

class Agent():
    def __init__(self, env):
        self.env = env
        self.network = Network(env.observation_space.shape, env.action_space.n)
        self.memory = PrioritizedMemory(2500)
        self.epsilon = 1.0

    def act(self, state):
        if random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return self.network.act(state)

    def train(self, state, action, next_state, reward, done, batch_size = 32):
        self.memory.push(state, action, reward, next_state, done)
        batch, importances, indices = self.memory.sample(batch_size)

        loss, errors = self.network.train(batch, importances**(1-self.epsilon))

        self.memory.set_priorities(indices, errors)

        if done: self.epsilon = max(.1, .99*self.epsilon)

        return loss

# Program

EPOCHS = 300
env = gym.make('CartPole-v0')
agent = Agent(env)
NAME = 'Prioritized-DDQN-CartPole-baseline-remasterized'

FirstMaxScore = None

writer = tf.summary.create_file_writer(f'tensorboard/{NAME}')

for episode in range(EPOCHS):
    loss_list = []
    reward_list = []
    done = False
    state = env.reset()
    while not done:
        if episode % (EPOCHS//10)== 0: env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        loss = agent.train(state, action, next_state, reward, done, batch_size=50)

        state = next_state
        reward_list.append(reward)
        loss_list.append(loss)

    episodeReward = np.sum(reward_list)
    episodeLoss   = np.mean(reward_list)

    if not FirstMaxScore and episodeReward == 200:
        FirstMaxScore = episode

    with writer.as_default():
        tf.summary.scalar('reward', episodeReward, step=episode)
        tf.summary.scalar('loss', episodeLoss, step=episode)
        tf.summary.scalar('epsilon', agent.epsilon, step=episode)

    print(f'episode {episode}/{EPOCHS} : reward {episodeReward} : loss {episodeLoss:.3f} : epsilon {agent.epsilon:.3f}')

print(f'First max reward at episode {FirstMaxScore}')
agent.network.policy_network.save(f'save_models/{NAME}e.h5')
