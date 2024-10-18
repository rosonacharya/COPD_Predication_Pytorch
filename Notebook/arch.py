#Define the model
import torch
import torch.nn as nn
import torch.optim as optim
class CODPmodel(nn.Module):
    def __init__(self,input_dim):
        super(CODPmodel,self).__init__()
        self.fc1=nn.Linear(input_dim,64)
        self.fc2=nn.Linear(64,32)
        self.fc3=nn.Linear(32,1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.sigmoid(self.fc3(x))
        return x