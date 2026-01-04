import torch
import matplotlib.pyplot as plt
from torch import nn
from x2paddle.project_convertor.pytorch.api_mapper import LinearModuleMapper

#1.数据处理
#1.1生成数据
torch.set_grad_enabled(True)
x = torch.linspace(-1,1,40).reshape(-1,1)
noise = torch.rand(x.size())
y = x*2 + noise

#1.2定义超参数
total_epoches = 1000       #迭代轮次
learning_rate = 0.01         #学习率
Loss = nn.MSELoss()         #损失函数

#2.学习过程
#2.1创建模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.fc = nn.Linear(1,1)
    def forward(self,x):
        return self.fc(x)
model = LinearModel()
#2.2训练模型
loss_list = []
epoch_list = []

optimizer = torch.optim.SGD(model.parameters(),   #指定需要优化的参数
                                               lr=learning_rate)       #指定学习率
for epoch in range(total_epoches):
    outputs = model(x)          #1、前向传播
    loss = Loss(outputs,y)      #2、计算损失
    optimizer.zero_grad()     #3、清空梯度
    loss.backward()               #4、反向传播
    optimizer.step()              #5、更新参数

    epoch_list.append(epoch)
    loss_list.append(loss.detach().item())

    if (epoch + 1) % 100 ==0:
        print(f'轮次/总轮次：{epoch}/{total_epoches}，损失值为：{loss:.4f}')

#2.3模型评估

#3.可视化
#3.1样本于模型
plt.figure(figsize=(16,7),dpi=100)
plt.subplot(1,2,1)
plt.title('Linear Regression')
plt.scatter(x.detach().numpy(),y.detach().numpy())
plt.plot(x.detach().numpy(),outputs.detach().numpy(),color = 'red')


#3.2训练过程
plt.subplot(1,2,2)
plt.title('Loss')
plt.plot(epoch_list,loss_list,color='red')
plt.tight_layout()
plt.show()









