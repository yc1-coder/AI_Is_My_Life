import torch

x = torch.tensor([1,2,3])

y = torch.ones(3)

z = x + y

if torch.backends.mps.is_available:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

x = x.to(device)
print(x)
y = y.to(device)
z = z.to(device)
print("使用设备",device)
print(z)