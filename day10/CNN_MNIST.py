import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# 1. 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 卷积层 1: 输入 1 通道，输出 32 通道，3x3 卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # 卷积层 2: 输入 32 通道，输出 64 通道，3x3 卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 2. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

# 3. 加载 MNIST 数据集
print("正在加载 MNIST 数据集...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 4. 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 5. 训练模型
def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    print(f'训练轮次：{epoch}, 平均损失：{total_loss / len(train_loader):.4f}, 准确率：{accuracy:.2f}%')


# 6. 测试模型
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    print(f'\n测试集 - 平均损失：{test_loss / len(test_loader):.4f}, 准确率：{accuracy:.2f}%\n')
    return accuracy


# 7. 可视化预测结果
def visualize_predictions(num_samples=10):
    model.eval()
    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)

    with torch.no_grad():
        example_data = example_data[:num_samples].to(device)
        output = model(example_data)
        predictions = output.cpu().max(1)[1].numpy()

    # 绘制图像
    fig = plt.figure(figsize=(15, 4))
    for idx in range(num_samples):
        ax = fig.add_subplot(2, num_samples // 2, idx + 1)
        img = example_data[idx].cpu().squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'预测：{predictions[idx]}\n真实：{example_targets[idx]}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('cnn_mnist_predictions.png', dpi=300)
    print("预测结果已保存为 cnn_mnist_predictions.png")
    plt.show()


# 8. 训练过程
print(f"使用设备：{device}")
print("=" * 60)

num_epochs = 100
accuracies = []

for epoch in range(1, num_epochs + 1):
    train(epoch)
    acc = test()
    accuracies.append(acc)

# 9. 绘制训练准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), accuracies, 'b-o', linewidth=2, markersize=8)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('CNN Training Accuracy on MNIST')
plt.grid(True)
plt.savefig('cnn_training_accuracy.png', dpi=300)
print("训练准确率曲线已保存为 cnn_training_accuracy.png")
plt.show()

# 10. 可视化一些预测示例
visualize_predictions(10)

# 11. 保存模型
torch.save(model.state_dict(), 'cnn_mnist_model.pth')
print("\n模型已保存为 cnn_mnist_model.pth")

print("\n" + "=" * 60)
print("训练完成！")
