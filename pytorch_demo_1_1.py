import torch
from torch import nn

#老三套
#这里拿卷积层举例子
class CNN(nn.Module):
    def __init__(self):
        #用来定义对应模型的结构
        super(CNN, self).__init__()
        #1、卷积层
        self.conv = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=0) #后两个参数有默认值，可以不填
        #in_channels:上一层图像通道数
        #out_channels: 输出图像通道数
        #kernel_size:卷积核大小
        #stride:步长
        #padding:填充长度

        #2、池话层
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)   #最大池化层
        self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)   #平均池化层
        # self.act = nn.ReLU()
    def forward(self,x):
        #定义数据流转过程，也就是正向传播过程
        x = torch.relu(self.conv(x))        #卷积+激活
        x = self.pool1(x)
        return x
#激活层，正常情况下会当做激活函数使用，这里只是为了方便，将激活层定义为函数
#另外可以当做神经网络使用nn.ReLU()
model = CNN()
print(model)

for param in model.parameters():
    print(param.shape)

#torch.Size([16, 3, 3, 3])
#最后两个是卷积核尺寸，16代表卷积核组数，3代表每组有多少个卷积核