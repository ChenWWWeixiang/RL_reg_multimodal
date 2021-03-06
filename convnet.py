import torch
import torch.nn as nn
import numpy as np
class Convnet(nn.Module):#TODO
    def __init__(self):
        super(Convnet, self).__init__()
        dummy=1
        self.bn=nn.BatchNorm3d(2)
        self.conv1=nn.Conv3d(2,32,(3,3,3))
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool3d((2,2,2),2)
        self.conv2=nn.Conv3d(32,48,(3,3,3))
        self.relu2 = nn.ReLU()
        self.pool2=nn.MaxPool3d((2,2,2),2)
        self.conv3=nn.Conv3d(48,96,(3,3,3))
        self.relu3 = nn.ReLU()
        self.pool3=nn.MaxPool3d((2,2,2),2)
        self.conv4=nn.Conv3d(96,128,(3,3,3))
        self.relu4 = nn.ReLU()
        self.conv5=nn.Conv3d(128,128,(3,3,3))
        self.relu5 = nn.ReLU()
        self.pool4=nn.MaxPool3d((2,2,1),(2,2,1))
        self.fc1=nn.Linear(73728,512)
        self.drop=nn.Dropout3d(0.5)
        self.relu6 = nn.ReLU()
        self.fc2=nn.Linear(512,128)
        self.relu7 = nn.ReLU()
        self.fc3=nn.Linear(128,64)
        self.relu8 = nn.ReLU()
        self.fc4=nn.Linear(64,9)

    def forward(self,crop_fixed,crop_moving):
        input=torch.Tensor(np.concatenate([crop_fixed,crop_moving],1)).float().cuda()
        x=self.bn(input)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        bn = self.pool4(x)
        fc1 = self.relu6(self.drop(self.fc1(bn)))
        fc2 = self.relu7(self.fc2(fc1))
        fc3 = self.relu78(self.fc3(fc2))
        Q_pre = self.fc4(fc3)
        return Q_pre