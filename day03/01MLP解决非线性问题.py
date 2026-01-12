import torch
from torch import nn
from torch import optim
from torchmetrics import MeanSquaredError,R2Score

#1、数据处理
#1.1 生成数据
x = torch.linspace(0,1,10).reshape(-1,1)
y = x ** 2 - 0.5 * x + 1.5625

#2.学习过程
#2.1 定义模型
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
optimizer = optim.Adam(model.parameters(),lr=0.01)
criterion = nn.MSELoss()       #回归模型，使用平均平方误差

#2.2模型训练
for epoch in range(10000):
    model.train()
    optputs = model(x)
    loss = criterion(optputs,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 500 == 0:
        print(f'epoch:{epoch},对应的损失值为:{loss:.4f}')

#2.3模型评估
model.eval()
outputs = model(x)

mse = MeanSquaredError()
r2 = R2Score()
print(f'mse:{mse(outputs,y)}')
print(f'r2:{r2(outputs,y)}')

#2.4模型保存
torch.save(model.state_dict(),'./my_model_params.pth')
print("模型已保存")



