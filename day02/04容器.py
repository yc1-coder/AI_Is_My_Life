#torch里面的容器专门管理神经网络层
#1.Sequential:序列容器
import torch
from torch import nn

class Net1(nn.Module):
    def __init__(self):
        super(Net1,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1,20,5), #卷积层
            #这里需要使用激活层的神经网络层的形式表示
            nn.ReLU(), #激活层
            nn.Conv2d(20,64,5), #卷积层
            nn.ReLU() #激活层
        )
    def forward(self,x):
        x = self.block(x)
        return x
net = Net1()
print(net)

for param in net.parameters():
    print(param.shape)

#2.ModuleList:列表容器
class Net2(nn.Module):
    def __init__(self):
        super(Net2,self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(3,10), #第一层
            nn.Linear(10,2), #二分类
            nn.Linear(10,3), #三分类
        ])
    def forward(self,x):
        x = self.layers[0](x)     #处理逻辑

        #处理逻辑不用较真，实际业务需要根据情况来处理
        feature_value = x[:,0]  #处理第一个维度的特征
        if feature_value.mean() > 0:
            x = self.layers[1](x)
        else:
            x = self.layers[2](x)
        return x
net = Net2()
print(net)

for param in net.parameters():
    print(param.shape)


input_data = torch.randn(5,3)
output = net(input_data)
print(output.shape)
print(output)