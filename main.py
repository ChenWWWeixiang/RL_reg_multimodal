import torch
import torch.nn as nn
import numpy as np
import glob
import random
import SimpleITK as sitk
BATCH_SIZE = 32
EXP_STEPS=1000
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
NUM_EPISODE=400
SAVE_PATH=''
N_ACTIONS=6
T1_path='/mnt/data2/pan_cancer/T1'
T2_path='/mnt/data2/pan_cancer/resampleT2'
class Env():
    def __init__(self,fixed_path,moving_path):
        dummy=1
        self.action_define=range(6)

    def takeAction(self,moving_img,action):#TODO
        return moving_img
    def reset(self,idx):#TODO
        fixed, moving=0,0
        return  fixed,moving

class Convnet(nn.Module):#TODO
    def __init__(self):
        super(Convnet, self).__init__()
        dummy=1
    def forward(self,crop_fixed,crop_moving):
        Q_pre=1
        return Q_pre

class Metricnet():#TODO
    def __init__(self):
        #super(Metricnet, self).__init__()
        dummy=1
    def forward(self, crop_fixed, crop_moving):
        r = 1
        done=0
        return r,done

class Agent():
    def __init__(self,memory_size=1000,crop_size=352,z_size=32,lr=LR):
        self.dqn,self.target=Convnet().cuda(),Convnet().cuda()
        self.crop_size=crop_size
        self.z_size=z_size
        self.memory_size=memory_size
        self.mb_pool=np.zeros(memory_size,crop_size,crop_size,z_size)
        self.maft_pool = np.zeros(memory_size, crop_size, crop_size, z_size)
        self.reward_pool = np.zeros(memory_size, 1)
        self.action_pool = np.zeros(memory_size, 1)
        self.count=0
        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
    def setFixed(self,fixed):
        self.fixed=fixed
    def chooseAction(self,m):
        m = torch.unsqueeze(torch.Tensor(m).cuda().float(), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            Q_pre = self.dqn(self.fixed,m)
            action = torch.argmax(Q_pre)[1].cpu().numpy()
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
        return action
    def saveState(self,m_bf,a,reward,m_aft):
        if self.count==self.memory_size:
            self.flushMem()
        self.mb_pool[self.count,...]=m_bf
        self.reward_pool[self.count, ...] = reward
        self.maft_pool[self.count, ...] = m_aft
        self.action_pool[self.count, ...] = a
        self.count+=1
    def flushMem(self):
        self.count=0
    def saveCKPT(self,path):
        torch.save(self.dqn.state_dict(),path)
    def learnDQN(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target.load_state_dict(self.dqn.state_dict())
        self.learn_step_counter += 1
        # sample batch transitions
        sample_index = np.random.choice(self.memory_size, BATCH_SIZE)
        b_memory = self.mb_pool[sample_index]
        aft_memory = self.maft_pool[sample_index]
        r_memory = self.reward_pool[sample_index]
        a_memory = self.action_pool[sample_index]

        b_s = torch.Tensor(b_memory).float().cuda()
        b_a = torch.Tensor(a_memory.astype(int)).long().cuda()
        b_r = torch.Tensor(r_memory).float().cuda()
        b_aft = torch.Tensor(aft_memory).float().cuda()

        # q_eval w.r.t the action in experience
        q_eval = self.dqn(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target(b_aft).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Trainer():
    def __init__(self):
        dummy=1
        self.env=Env(T1_path,T2_path)
        self.r_metric=Metricnet()
        self.agent=Agent()
        self.fixedfile_list=glob.glob(T1_path)
    def experience(self,i_episode,exp_steps):
        idx=random.randint(len(self.fixedfile_list))
        fixed,moving = self.env.reset(idx)
        ep_r = 0
        for i in range(exp_steps):
            self.agent.setFixed(fixed)
            a = self.agent.chooseAction(moving)# a is a numpy
            # take action
            moving_new = self.env.takeAction(moving,a)
            r,done=self.r_metric.forward(fixed,moving_new)
            self.agent.saveState(moving, a, r, moving_new)
            ep_r += r
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r / i))
                return done
            moving = moving_new
        print('Ep: ', i_episode,'| Ep_r: ', round(ep_r/exp_steps))
        return done
    def learn(self):
        self.agent.learnDQN()

    def gogogo(self):
        for i in range(NUM_EPISODE):
            done=self.experience(i,EXP_STEPS)
            if done:
                break
            self.learn()
        self.agent.saveCKPT(SAVE_PATH)

t=Trainer()
t.gogogo()