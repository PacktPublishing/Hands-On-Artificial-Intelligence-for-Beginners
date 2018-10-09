''' Run a Deep Q Network to Play Atari '''

import cv2
import sys
import gym
import numpy as np

from deepQ import deepQ

class AtariGame:
    '''Class to play the spaceinvaders atari game'''
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        self.env.reset()
        self.actions = self.env.action_space.n
        self.deepQ = deepQ(self.actions)
        self.action0 = 0

    def preprocess(self, observation):
        ''' Preprocess the video input. Taken from @floodsung's adapation of DQN'''
	    obs = cv2.cvtColor(cv2.resize(obs, (84, 110)), cv2.COLOR_BGR2GRAY)
	    obs = obs[26:110,:]
	    ret, obs = cv2.threshold(obs, 1,255,cv2.THRESH_BINARY)
	    return np.reshape(obs, (84,84,1))

    def run_atari(self):
        obs, rew, ter, info = self.env.step(self.action0)
        obs = self.preprocess(obs)
        self.deepQ.initialState(obs)
        self.deepQ.currentState = np.squeeze(self.deepQ.currentState)

        while True:
            action = self.deepQ.select()
            action_max = np.argmax(np.array(action))
            nextObs, reward, terminal, info = self.env.step(action_max)
            if terminal:
                nextObservation = self.env.reset()
                nextObservation = self.preprocess(nextObs)
                self.deepQ.er_replay(nextObs, action, reward, terminal)

if __name__ == '__main__':
    a = AtariGame()
    a.run_atari()
