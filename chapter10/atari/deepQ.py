''' Deep Q Network to Play Atari. Adapated from DeepQ by @floodsung '''

import tensorflow as tf
import numpy as np
import random

from collections import deque

class deepQ:
	def __init__(self, action):

		## Initialize the network's variables
		self.replayBuffer = deque()
		self.action = action
		self.timeStep = 0
		self.starting_ep = 1.0
		self.ending_ep = 0.1

		self.observe = 50000
		self.explore = 1000000

		## Initialize the base network
		self.inputVal, self.QValue = self.deepQarchitecture()

		## Initialize the target network
		self.inputValT, self.QValueT = self.deepQarchitecture()

		## Initialize the Training Procedure
		self.TrainingProcedure()

		## Initial a saver for the network's variables, a session, and then initialize
		## all of the network's variables
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())


	def deepQarchitecture(self):
		''' Architecture of the Deep Q Network. Creates the weight and
		bias factors for each network layer, and then creates the network
		layer itself. '''

		## Network input layer
		input_layer = tf.placeholder("float",[None,84,84,4])

		## Convolutional layers
		conv1 = tf.layers.conv2d(inputs = input_layer, filters = 32, kernel_size=[8, 8],
			padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer,
            use_bias=True, bias_initializer=initializer, bias_regularizer=regularizer,
            activation=tf.nn.relu
			)

		conv2 = tf.layers.conv2d(inputs = conv1, filters = 32, kernel_size=[4, 4]
		 	padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer,
            use_bias=True, bias_initializer=initializer, bias_regularizer=regularizer,
            activation=tf.nn.relu
			)

		conv3 = tf.layers.conv2d(inputs = conv2, filters = 64, kernel_size=[3, 3]
		 	padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer,
            use_bias=True, bias_initializer=initializer, bias_regularizer=regularizer,
            activation=tf.nn.relu
			)

		## Flatten the last convolutional layer
		conv3_shape = conv3.get_shape().as_list()
		conv3_flat = tf.reshape(conv3,[-1,3136])

		## Fully Connected Layer
		fc = tf.layers.dense(inputs = conv3_flat, units = 512, kernel_initializer=initializer,
			kernel_regularizer=regularizer use_bias=True, bias_initializer=initializer,
			bias_regularizer=regularizer, activation=tf.nn.relu
			)

		## Output Q Value
		QValue = tf.layers.dense(inputs = fc, units = self.action, kernel_initializer=initializer,
			kernel_regularizer=regularizer use_bias=True, bias_initializer=initializer,
			bias_regularizer=regularizer, activation = None)

		return input_layer, QValue

	def TrainingProcedure(self):
		''' Creates the training procedure for each time step in the
		overall training process '''

		self.actionInput = tf.placeholder("float",[None, self.action])
		self.yInput = tf.placeholder("float", [None])
		Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		self.trainStep = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)


	def train(self):
		''' Training procedure for the Q Network'''

		minibatch = random.sample(self.replayBuffer, 32)
		stateBatch = [data[0] for data in minibatch]
		actionBatch = [data[1] for data in minibatch]
		rewardBatch = [data[2] for data in minibatch]
		nextBatch = [data[3] for data in minibatch]

		batch = []
		qBatch = self.QValueT.eval(feed_dict = {self.inputValT: nextBatch})
		for i in range(0, 32):
			terminal = minibatch[i][4]
			if terminal:
				batch.append(rewardBatch[i])
			else:
				batch.append(rewardBatch[i] + self.gamma * np.max(qBatch[i]))

		self.trainStep.run(feed_dict={
			self.yInput : batch,
			self.actionInput : actionBatch,
			self.inputVal : stateBatch
			})

		## Save the network on specific iterations
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, './savedweights' + '-atari', global_step = self.timeStep)

	## Experience Replay
	def er_replay(self, nextObservation, action, reward, terminal):
		newState = np.append(nextObservation, self.currentState[:,:,1:], axis = 2)
		self.replayMemory.append((self.currentState, action, reward, newState, terminal))
		if len(self.replayBuffer) > 40000:
			self.replayBuffer.popleft()
		if self.timeStep > self.explore:
			self.trainQNetwork()

		self.currentState = newState
		self.timeStep += 1

	def select(self):
		## Select a Q Value from the Base Q Network
		QValue = self.QValue.eval(feed_dict = {self.iputVal:[self.currentState]})[0]
		## Initialize actions as zeros
		action = np.zeros(self.action)
		action_index = 0
		## If this timestep is the first, start with a random action
		if self.timeStep % 1 == 0:
			##
			if random.random() <= self.starting_ep:
				a_index = random.randrange(self.action)
				action[a_index] = 1
			else:
				action_index = np.argmax(QValue)
				action[action_index] = 1
		else:
			action[0] = 1

		## Anneal the value of epsilon
		if self.starting_ep > self.ending_ep and self.timeStep > self.observe:
			self.starting_ep -= (self.starting_ep - self.ending_ep) / self.explore

		return action

	def initial_state(self, obs):
		self.currentState = np.stack((obs, obs, obs, obs), axis = 2)
