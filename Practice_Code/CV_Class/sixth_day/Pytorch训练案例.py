import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*16*16,10)


    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x

model=SimpleCNN()
print(model)

