import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def load_trained_model(model_path='cat_dog_classifier.pth'):
    """加载训练好的猫狗分类模型"""
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def classify_cat_or_dog(image_path):
    """对单张图片进行猫狗分类"""
    model, device = load_trained_model()
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 预测
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
    # 返回结果
    class_names = ['cat', 'dog']
    result = {
        'class': class_names[predicted_class],
        'confidence': probabilities[0][predicted_class].item(),
        'all_probabilities': {
            'cat': probabilities[0][0].item(),
            'dog': probabilities[0][1].item()
        }
    }
    return result

# 使用示例
result = classify_cat_or_dog('images.jpeg')
print(f"预测类别: {result['class']}")
print(f"置信度: {result['confidence']:.4f}")
