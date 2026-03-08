from torch.utils.data import DataLoader
from torchvision import datasets,transforms

#1.数据处理
#1.1 数据预处理
transform = transforms.Compose([
    #1、先调整图片大小
    transforms.Resize((28,28)),
    #2、数据增强和随机裁剪
    transforms.RandomHorizontalFlip(p=0.2),
    #3、转换成张量
    transforms.ToTensor(),
    #4、进行归一化
    transforms.Normalize(mean = (0.5,),std = (0.5,))
])

# 1.2加载数据集
train_dataset = datasets.MNIST(root='data',      #数据集存放路径
                                                     train = True,         #是否为训练集
                                                     transform = transform,
                                                     download = True,   #如果为True,自动下载;如果有，则取消下载
                               )
test_dataset = datasets.MNIST(root='data',                   #数据集的存放路径
                                                      train = False,                    #是否为训练集
                                                      transform = transform,    #数据预处理
                                                      download=True
                              )

#1.3数据封装
#封装到DataLoader里面，变成一个生成器，每次返回一个batch_size大小的数据
train_loader = DataLoader(train_dataset,      #封装的原数据集
                          batch_size = 50,                    #批大小
                          shuffle = True,                      #是否打乱数据集(训练集打乱)
                          )
test_loader = DataLoader(test_dataset,        #封装的原数据
                         batch_size= 50,                     #批大小
                         shuffle = False,
                         )

#2.获取数据的两种方法
#2.1 正常获取
for batch_data, batch_labels in train_loader:
    print(batch_data.shape)
    print(batch_labels.shape)
    break

#2.2 使用enumerate
#和生成器或迭代器结合使用，可以获得对应数据的索引
for idx,(batch_data,batch_labels) in enumerate(train_loader):
    if idx < 3:
        print(f'第{idx}个批次')
        print(f'数据形状{batch_data.shape}')
        print(f'标签形状{batch_labels.shape}')
    else:
        break





