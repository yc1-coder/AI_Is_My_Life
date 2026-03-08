import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader

#1、数据预处理与加载（核心：归一化+数据增强）
transform = transforms.Compose([
    transforms.Resize((224,224)),       #统一图像尺寸
    transforms.RandomHorizontalFlip(),  #简单数据增强
    transforms.ToTensor(),              #转换为张量
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])   #归一化
])

#替换为你的猫狗数据集路径（结构：train/cat,train/dog,val/cat,cal/dog）
train_dataset = datasets.ImageFolder('cat-dog_data/train',transform=transform)
val_dataset = datasets.ImageFolder('cat-dog_data/validation',transform=transform)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)

#2.定义模型（复用预训练RestNet18,修改最后一层适配而分类）
model = models.resnet18(pretrained=True)
#冻结特征提取层（加速训练）
for param in model.parameters():
    param.requires_grad = False
#替换全连接层，输出2类（猫/狗）
model.fc = nn.Linear(model.fc.in_features,2)
model = model.to('mps' if torch.backends.mps.is_available() else 'cpu')

#3.定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(),lr=0.001)

#4.训练与验证核心逻辑
def train_one_epoch(model,loader,optimizer,criterion):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device),labels.to(device)
        optimizer.zero_grad()   #梯度清零
        outputs = model(images) #前向传播
        loss = criterion(outputs,labels)  #计算损失
        loss.backward()         #反向传播
        optimizer.step()        #更新参数
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model,loader,criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():      #验证阶段禁用梯度
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            total_loss += loss.item()
            _,preds = torch.max(outputs,1)   #获取预测类别
            correct += torch.sum(preds == labels.data)
    accuracy = correct.float() / len(loader.dataset)
    return total_loss / len(loader),accuracy

#启动训练（仅训练前5轮示例）
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

for epoch in range(99):
    train_loss = train_one_epoch(model,train_loader,optimizer,criterion)
    val_loss, val_acc = validate(model,val_loader,criterion)
    print(f'Epoch{epoch+1}:训练损失={train_loss:.4f},验证损失={val_loss:.4f},验证准确率={val_acc:.4f}')

#保存模型
torch.save(model.state_dict(),'cat_dog_classifier.pth')








