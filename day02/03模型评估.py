
import os
# 解决 OMP 错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, MeanSquaredError, R2Score

#1.分类模型
pred_y = torch.tensor([1,0,0,1,0])
true_y = torch.tensor([1,1,0,1,0])

print(f'准确率:{Accuracy(task="binary")(pred_y,true_y):.2f}')
print(f'F1:{F1Score(task="binary")(pred_y,true_y):.2f}')
print(f'查准率:{Precision(task="binary")(pred_y,true_y):.2f}')
print(f'查全率:{Recall(task="binary")(pred_y,true_y):.2f}')

#2.回归模型
pred_y = torch.tensor([15.2,14.3,11.2,9.5]).reshape(-1,1)
true_y = torch.tensor([14.5,13.2,10.5,9.2]).reshape(-1,1)
mse = MeanSquaredError()
mse.update(pred_y,true_y)
print(f'MSE:{mse.compute():.2f}')
print(f'R2:{R2Score()(pred_y,true_y):.2f}')