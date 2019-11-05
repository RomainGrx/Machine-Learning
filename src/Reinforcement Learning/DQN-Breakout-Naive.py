#!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Wed Aug 14 15:30:25 2019
#
#@author: romaingraux
#"""


import gym
import numpy as np
import sys
sys.path.insert(0, '/Users/romaingraux/Documents/Python/Machine-Learning/res/prog')
import useful as u
import selflayers as l
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import random
from time import sleep
modelpath = '../../res/models/DQN-Breakout-Naive.h5'
tensorboardpath = '/Users/romaingraux/Documents/Python/Machine-Learning/res/tensorboard/DQN-Breakout/run1'

def Convolution_activate(input, neurons_in, neurons_out, shape, pooling=False, name = 'Convolution'):
    with tf.name_scope(name):
        w = tf.get_variable('W', [*shape, neurons_in, neurons_out], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.Variable(tf.constant(0.1, shape = [neurons_out]), name = 'B')
        conv = tf.nn.conv2d(input, w, strides = [1,1,1,1], padding = 'SAME')
        act = tf.nn.relu(conv + b)
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        pool = tf.nn.max_pool(act, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') if pooling else act
    return pool

def Fully_connected(input, neurons_in, neurons_out, name = 'Fully conncected', activation = None):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([neurons_in, neurons_out], stddev=0.1), name = 'W')
        b = tf.Variable(tf.constant(0.1, shape = [neurons_out]), name = 'B')
        mul = tf.matmul(input, w) + b
        if activation == 'relu':
            act = tf.nn.relu(mul)
        elif activation == 'softmax' :
            act = tf.nn.softmax(mul)
        else : 
            act = mul
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
    return act

class DQN():
    
    def __init__(self, state_size, action_size, learning_rate, name='DQN'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 1], name="actions_")
            
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                    name = 'batch_norm1')
            
            self.conv1_out = tf.nn.relu(self.conv1_batchnorm, name="conv1_out")
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                 kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.relu(self.conv2_batchnorm, name="conv2_out")
            
            
            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                 kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                                 name = "conv3")
        
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')

            self.conv3_out = tf.nn.relu(self.conv3_batchnorm, name="conv3_out")
            ## --> [3, 3, 128]
            
            
            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]
            
            
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.sigmoid,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc1")
            
            
            self.output = tf.layers.dense(inputs = self.fc, 
                                          kernel_initializer='random_uniform',
                                          units = 3, 
                                          activation=None)

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
        with tf.name_scope('Loss'):
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

class FrameProcessor():

    def __init__(self, frame_height = 84, frame_width = 84):
        with tf.variable_scope("FrameProcessor"):
            self.frame_height = frame_height
            self.frame_width = frame_width
            self.frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.processed = tf.image.rgb_to_grayscale(self.frame)
            self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
            self.processed = tf.image.resize_images(self.processed, 
                                                    [self.frame_height, self.frame_width], 
                                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            tf.summary.image('input', self.processed, 30)
    def __call__(self, session, frame):
        return session.run(self.processed, feed_dict={self.frame:frame})[:,:,0]
    
class StackFrames():
    
    def __init__(self, stack_size = 3, frame_height = 84, frame_width = 84):
        self.stack_size = stack_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.stack = deque([np.zeros((self.frame_height,self.frame_width), dtype=np.uint8) for i in range(stack_size)], maxlen=stack_size) 
        
    def __call__(self, processed_frame, is_new_episode):
        if is_new_episode:
            self.stack = deque([np.zeros((self.frame_height,self.frame_width), dtype=np.uint8) for i in range(self.stack_size)], maxlen=self.stack_size) 
            self.stack.append(processed_frame)
            self.stack.append(processed_frame)
            self.stack.append(processed_frame)
            stacked_state = np.stack(self.stack, axis=2)
        else:
            self.stack.append(processed_frame)
            stacked_state = np.stack(self.stack, axis=2)
            
        return stacked_state
    

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size = 64):
        buffer_size = len(self.buffer)
        batch_size = min(batch_size, buffer_size)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]
    
class Agent():
    
    def __init__(self, episodes = 500, env_name = 'BreakoutDeterministic-v4', processed_shape = (84, 84), stack_size = 3, epsilon_start = 1.0, epsilon_stop = 0.01, learning_rate = 0.0002, gamma = 0.95, memory_size = 10000, batch_size = 64):
        self.possibles_actions = [[0],[1],[2],[3]]
        self.episodes = episodes
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.processed_shape = processed_shape
        self.stack_size = stack_size
        self.FrameProcessor = FrameProcessor(frame_height = self.processed_shape[0], frame_width = self.processed_shape[1])
        self.StackFrames = StackFrames(stack_size = self.stack_size, frame_height = self.processed_shape[0], frame_width = self.processed_shape[1])
        self.Memory = Memory(max_size = memory_size)
        self.batch_size = batch_size
        self.stack = self.StackFrames.stack
        self.epsilon = 1.0
        self.epsilon_start = epsilon_start
        self.epsilon_stop = epsilon_stop
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.DQN = DQN(state_size = [*self.processed_shape, self.stack_size], action_size = self.action_size, learning_rate = self.learning_rate)
        self.loss = 666
        self.current_reward = 0
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Rewards", self.current_reward)
        self.write_op = tf.summary.merge_all()
        
    def act(self, state, episode, sess):
        rand = random.random()
#        self.epsilon = self.epsilon_stop + (self.epsilon_start - self.epsilon_stop) * np.exp(-0.1 * episode / self.episodes)
        self.epsilon = 1 - episode / self.episodes
        if rand < self.epsilon : 
            action = self.env.action_space.sample()
        else :
            action = sess.run(self.DQN.output, feed_dict = {self.DQN.inputs_: state.reshape((1, *state.shape))})
            choice = np.argmax(action)
            action = int(choice)
        return self.possibles_actions[action]
    
    def plot_results(self, reward_list):
        plt.plot((np.arange(len(reward_list))), reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episodes')
        plt.show()
    
    def train(self, render = False):
        # Saver will help us to save our model
        saver = tf.train.Saver()
#        self.DQN = u.modelsave(modelpath, self.DQN)
        
        sess = tf.Session()
        self.write_op = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(tensorboardpath, sess.graph)
#        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
#        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(self.writer, config)
        # Initialize the variables
        
        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0

        # Init the game
        episodes_mean_rewards = []
        
        for episode in range(self.episodes):
            step = 0
            
            frame = self.env.reset()
            state = self.FrameProcessor(sess, frame)
            
            episode_rewards = []
            
            state = self.StackFrames(state, True)
            
            done = False
            
            while not done:
                if render : self.env.render()
                step += 1
                
                decay_step +=1
                
                action = self.act(state, episode, sess)
                next_frame, reward, done, info = self.env.step(action)
                next_frame = self.FrameProcessor(sess, next_frame)
                self.current_reward = reward
                
                episode_rewards.append(reward)

                # If the game is finished
                if done == True:
                    
                    next_frame = np.zeros(self.processed_shape, dtype=np.int)
                    next_state= self.StackFrames(next_frame, True)
                    

                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}/{}'.format(episode, self.episodes),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(self.loss),
                              'Explore P: {:.4f}'.format(self.epsilon))

                    self.Memory.add((state, action, reward, next_state, done))

                else:
                    
                    next_state = self.StackFrames(next_frame, False)
                    
#                    plt.figure()
#                    plt.subplot(131)
#                    plt.imshow(next_state[:,:,0])
#                    plt.subplot(132)
#                    plt.imshow(next_state[:,:,1])
#                    plt.subplot(133)
#                    plt.imshow(next_state[:,:,2])
#                    plt.show()

                    self.Memory.add((state, action, reward, next_state, done))
                    
                    state = next_state


                ### LEARNING PART            
                # Obtain random mini-batch from memory
                batch = self.Memory.sample(self.batch_size)
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                 # Get Q values for next_state 
                Qs_next_state = sess.run(self.DQN.output, feed_dict = {self.DQN.inputs_: next_states_mb})
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + self.gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])
                plt.imshow(states_mb[0])

                self.loss, _, summary = sess.run([self.DQN.loss, self.DQN.optimizer, self.write_op],
                                    feed_dict={self.DQN.inputs_: states_mb,
                                               self.DQN.target_Q: targets_mb,
                                               self.DQN.actions_: actions_mb})


                self.writer.add_summary(summary, episode)
                self.writer.flush()
            episodes_mean_rewards.append(np.mean(episode_rewards))
    
        self.plot_results(episodes_mean_rewards)
        u.modelsave(modelpath, self.DQN)
  
      
tf.reset_default_graph() 
tf.summary.FileWriterCache.clear() 

 
Atari = Agent(episodes = 300, stack_size = 4)
try : 
    Atari.train(render=True)
finally :
    Atari.env.close()
#env = gym.make('MountainCar-v0')
#print(env.observation_space.shape)





    
        