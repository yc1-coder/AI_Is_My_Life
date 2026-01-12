from torchvision import datasets,transforms
transforms.Compose([
    #1、先调整图片大小
    transforms.Resize((28,28)),
    #2、数据增强和随机裁剪
    transforms.RandomHorizontalFlip(p=0.2),
    #3、转换成张量
    transforms.ToTensor(),
    #4、进行归一化
    transforms.Normalize(mean = (0.5,),std = (0.5,))
])