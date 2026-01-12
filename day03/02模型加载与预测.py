#1.加载的时候，要把参数对应的模型写下来或者导入进来
import torch
from torch import nn
from torch import optim
from torchmetrics import MeanSquaredError,R2Score
import matplotlib.pyplot as plt
#1、加载
x = torch.linspace(0,1,10).reshape(-1,1)
y = x ** 2 - 0.5 * x + 1.5625

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #定义多层感知机
        self.block = nn.Sequential(
            nn.Linear(1,2),
            nn.Sigmoid(),
            nn.Linear(2,1)
        )
    def forward(self,x):
        x = self.block(x)
        return x
model = Net()

#加载参数
model.load_state_dict(torch.load('./my_model_params.pth'))

test_x = torch.linspace(0,1,100).reshape(-1,1)
pred_test_y = model(test_x).detach().numpy()

#3.可视化
plt.figure(figsize=(16,9),dpi=80)
plt.scatter(x,y,color = 'blue',marker='*',label = 'ture data')
plt.plot(test_x,pred_test_y,color = 'red',label = 'pred data')
plt.legend()
plt.show()