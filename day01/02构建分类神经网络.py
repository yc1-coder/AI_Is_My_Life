import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.fc1 = nn.Linear(10,5) #输入10个特征，输出5个特征
        self.fc2 = nn.Linear(5,2) #输入5个特征，输出2个特征

    def forward(self,x):
        # x = torch.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)      #使用ReLU激活函数
        x = self.fc2(x)
        return x
