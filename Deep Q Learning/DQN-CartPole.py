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
class Memory():
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def push(self, *memory_state):
        self.memory.append(Memory_State(*memory_state))

    def sample(self, size):
        sample_memory = random.sample(self.memory, min(len(self), size))
        batch = Memory_State(*zip(*sample_memory))
        return batch

    def __len__(self):
        return len(self.memory)

# Network class

class Network():
    def __init__(self, input_shape, output_n):
        self.input_shape = input_shape
        self.output_n = output_n
        self.policy_network = self.get_model()
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.gamma = .97

    def get_model(self):
        model = tf.keras.Sequential([
                                     tf.keras.layers.Dense(100, input_shape=self.input_shape, activation=tf.nn.relu),
                                     tf.keras.layers.Dense(self.output_n, activation=None)

        ])
        return model

    def act(self, state):
        q_values = self.model(np.expand_dims(state, 0))
        action = np.argmax(q_values)
        return action

    def train(self, batch):

        state_batch = tf.convert_to_tensor(batch.state)
        action_batch = tf.convert_to_tensor(batch.action)
        reward_batch = tf.convert_to_tensor(batch.reward)
        next_state_batch = tf.convert_to_tensor(batch.next_state)
        done_batch = tf.convert_to_tensor(batch.done)

        variables = self.model.trainable_variables

        q_next_states = self.model(next_state_batch)
        q_target = tf.where(done_batch, reward_batch, reward_batch + self.gamma * tf.math.reduce_max(q_next_states, axis=1))

        with tf.GradientTape() as tape:
            tape.watch(variables)

            q_states = self.model(state_batch)

            action_one_hot = tf.one_hot(action_batch, depth=self.output_n)
            q_values_actions = tf.math.reduce_sum(tf.multiply(q_states, action_one_hot) , axis=1)

            loss = tf.reduce_mean(tf.square(q_values_actions - q_target))


        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

# Agent class

class Agent():
    def __init__(self, env):
        self.env = env
        self.network = Network(env.observation_space.shape, env.action_space.n)
        self.memory = Memory(20000)
        self.epsilon = 1.0

    def act(self, state):
        if random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return self.network.act(state)

    def train(self, state, action, next_state, reward, done, batch_size = 32):
        self.memory.push(state, action, reward, next_state, done)
        batch = self.memory.sample(batch_size)
        loss = self.network.train(batch)

        if done: self.epsilon = max(.1, .99*self.epsilon)

        return loss


# Program

EPOCHS = 300
env = gym.make('CartPole-v0')
agent = Agent(env)

writer = tf.summary.create_file_writer('tensorboard/DQN-CartPole-baseline')

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

    with writer.as_default():
        tf.summary.scalar('reward', np.sum(reward_list), step=episode)
        tf.summary.scalar('loss', np.mean(loss_list), step=episode)
        tf.summary.scalar('epsilon', agent.epsilon, step=episode)

    print(f'episode {episode}/{EPOCHS} : reward {np.sum(reward_list)} : loss {np.mean(loss_list):.3f} : epsilon {agent.epsilon:.3f}')

agent.network.model.save('save_models/DQN-CartPole-baseline.h5')
