# -*- coding: utf-8 -*-
"""
Данный модуль определяет структуру нейронной сети, 
прямое распространение сигнала,
оптимизируемую функцию

Архитектура нейронной сети - нейронная сеть Ворда.


10 Октября 2019

"""


import torch
import  torch.nn.functional as F
import torch.nn as nn

# Set loss function (mean absolute error)
def mape_loss(y_true, y_pred):
    r=y_true-y_pred
    r=torch.abs(r)
    r/=torch.abs(y_pred)
    r=torch.mean(r)
    return r

# get accuracy
def get_ac(mape_value):
    ac=round(float(100.0-100.0*mape_value),1)
    return ac
        
# create the forward pass function of deep neural network
class DNN(torch.nn.Module):
    def __init__(self, inp):
        super(DNN, self).__init__()
        # =====================================================================
        # block #1
        # =====================================================================
        
        i1=round(inp)
        o1=round(inp/8)
        
        self.fc1_1 = nn.Linear(i1,o1) # linear layer 1
        self.bn1_1 = nn.BatchNorm1d(o1) # normalization layer
        
        self.fc1_2 = nn.Linear(i1,o1) # linear layer 1
        self.bn1_2 = nn.BatchNorm1d(o1) # normalization layer
        
        self.fc1_3 = nn.Linear(i1,o1) # linear layer 1
        self.bn1_3 = nn.BatchNorm1d(o1) # normalization layer
                
        # =====================================================================
        # block #2
        # =====================================================================
        
        i2=o1
        o2=round(o1/2)
        
        self.fc2_1 = nn.Linear(i2,o2) # linear layer 1
        self.bn2_1 = nn.BatchNorm1d(o2) # normalization layer
        
        self.fc2_2 = nn.Linear(i2,o2) # linear layer 1
        self.bn2_2 = nn.BatchNorm1d(o2) # normalization layer
        
        self.fc2_3 = nn.Linear(i2,o2) # linear layer 1
        self.bn2_3 = nn.BatchNorm1d(o2) # normalization layer
        
        self.fc2_4 = nn.Linear(i2,o2) # linear layer 1
        self.bn2_4 = nn.BatchNorm1d(o2) # normalization layer
        
        self.fc2_5 = nn.Linear(i2,o2) # linear layer 1
        self.bn2_5 = nn.BatchNorm1d(o2) # normalization layer
        
        self.fc2_6 = nn.Linear(i2,o2) # linear layer 1
        self.bn2_6 = nn.BatchNorm1d(o2) # normalization layer
        
        self.fc2_7 = nn.Linear(i2,o2) # linear layer 1
        self.bn2_7 = nn.BatchNorm1d(o2) # normalization layer
        
        self.fc2_8 = nn.Linear(i2,o2) # linear layer 1
        self.bn2_8 = nn.BatchNorm1d(o2) # normalization layer
        
        self.fc2_9 = nn.Linear(i2,o2) # linear layer 1
        self.bn2_9 = nn.BatchNorm1d(o2) # normalization layer
        
        # =====================================================================
        # block #3
        # =====================================================================
        
        i3=round(o2*3)
        o3=round(o2/2)
        
        self.fc3_1 = nn.Linear(i3,o3) # linear layer 1
        self.bn3_1 = nn.BatchNorm1d(o3) # normalization layer
        
        self.fc3_2 = nn.Linear(i3,o3) # linear layer 1
        self.bn3_2 = nn.BatchNorm1d(o3) # normalization layer
        
        self.fc3_3 = nn.Linear(i3,o3) # linear layer 1
        self.bn3_3 = nn.BatchNorm1d(o3) # normalization layer
        
        # =====================================================================
        # block #4
        # =====================================================================
        
        in4=round(o3*3+inp)
        o4=1
        self.fc4_1 = nn.Linear(in4,o4)
  
        
    def forward(self, x):
        
        # block 1
        x1_1 = self.bn1_1(F.softplus(self.fc1_1(x)))
        x1_2 = self.bn1_2(F.logsigmoid(self.fc1_2(x)))
        x1_3 = self.bn1_3(torch.tanh(self.fc1_3(x)))
        
        # block 2
        x2_1 = self.bn2_1(F.softplus(self.fc2_1(x1_1)))
        x2_2 = self.bn2_2(F.logsigmoid(self.fc2_2(x1_1)))
        x2_3 = self.bn2_3(torch.tanh(self.fc2_3(x1_1)))
        
        x2_4 = self.bn2_4(F.softplus(self.fc2_4(x1_2)))
        x2_5 = self.bn2_5(F.logsigmoid(self.fc2_5(x1_2)))
        x2_6 = self.bn2_6(torch.tanh(self.fc2_6(x1_2)))
        
        x2_7 = self.bn2_7(F.softplus(self.fc2_7(x1_3)))
        x2_8 = self.bn2_8(F.logsigmoid(self.fc2_8(x1_3)))
        x2_9 = self.bn2_9(torch.tanh(self.fc2_9(x1_3)))
        
        # block 3
        x3_1 = torch.cat((x2_1,x2_2,x2_3),1)
        x3_2 = torch.cat((x2_4,x2_5,x2_6),1)
        x3_3 = torch.cat((x2_7,x2_8,x2_9),1)
        
        x3_1 = self.bn3_1(F.softplus(self.fc3_1(x3_1)))
        x3_2 = self.bn3_2(F.logsigmoid(self.fc3_2(x3_2)))
        x3_3 = self.bn3_3(torch.tanh(self.fc3_2(x3_3)))
        
        # block 4
        x4_1 = torch.cat((x3_1,x3_2,x3_3,x),1)
        x = F.relu(self.fc4_1(x4_1))        
        
        return x