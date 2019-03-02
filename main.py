import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000

class Env():
    def __init__(self,fixed_path,moving_path):
        dummy=1
        self.action_define=range(6)
    def reset(self,fixed_img):
        moving_sampled = 1
        return moving_sampled
    def takeaction(self,moving_img,action):
        return moving_img

class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        dummy=1
    def forward(self,crop_fixed,crop_moving):
        Q_pre=1
        return Q_pre

class Metricnet(nn.Module):
    def __init__(self):
        super(Metricnet, self).__init__()
        dummy=1
    def forward(self, crop_fixed, crop_moving):
        r = 1
        return r

class Agent():
    def __init__(self,memory_size=1000,crop_size=352,z_size=32,lr=LR):
        self.dqn,self.target=Convnet(),Convnet()
        self.crop_size=crop_size
        self.z_size=z_size
        self.memory_size=memory_size
        self.mb_pool=np.zeros(memory_size,crop_size,crop_size,z_size)
        self.maft_pool = np.zeros(memory_size, crop_size, crop_size, z_size)
        self.reward_pool = np.zeros(memory_size, 1)
        self.count=0
        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)
        self.loss_func = Metricnet()
    def setFixed(self,fixed):
        self.fixed=fixed
    def chooseAction(self,m):
        Q_pre=self.dqn(self.fixed,m)
        action_idx=torch.argmax(Q_pre)
        return action_idx
    def saveState(self,m_bf,reward,m_aft):
        self.mb_pool[self.count,...]=m_bf
        self.reward_pool[self.count, ...] = reward
        self.maft_pool[self.count, ...] = m_aft
        self.count+=1
    def flushMem(self):
        self.count=0
    def learnDQN(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target.load_state_dict(self.dqn.state_dict())
        self.learn_step_counter += 1

