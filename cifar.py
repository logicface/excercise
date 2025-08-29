import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
if torch.cuda.is_available():
    device = torch.device("cuda")
    # print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    # print(f"GPU数量: {torch.cuda.device_count()}")
    # print(f"CUDA版本: {torch.version.cuda}")
    torch.cuda.empty_cache()

import tensorflow as tf
import torchvision
from torch.utils.data import Dataset, DataLoader
import os

data_path = './data'
cifar10_path = os.path.join(data_path, 'cifar-10-batches-py')

def check_local_files():
    """检查CIFAR-10数据文件是否存在"""
    if os.path.exists(cifar10_path) and os.path.isdir(cifar10_path):
        # 检查是否包含必要的文件
        required_files = ['batches.meta', 'data_batch_1', 'data_batch_2', 
                         'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
        for file in required_files:
            if not os.path.exists(os.path.join(cifar10_path, file)):
                return False
        return True
    return False

# 使用torchvision下载CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


if check_local_files():
    print("检测到本地CIFAR-10数据文件，直接加载...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
else:
    print("未检测到本地数据，开始下载CIFAR-10数据集...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

# 数据预处理保持不变
# trainset.data = trainset.data / 255.0
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

# testset.data = testset.data / 255.0
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

# CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# print(len(trainset))

# plt.figure(figsize=(20, 10))
# for i in range(20):
#     plt.subplot(2,10,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     # 修正：trainset[i]返回(image, label)元组，需要取第一个元素image
#     image, label = trainset[i]
#     plt.imshow(image.permute(1, 2, 0))  # 调整通道顺序：从(C,H,W)到(H,W,C)
#     plt.xlabel(classes[label])
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 改进的网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # 根据网络结构计算全连接层输入大小
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2 * 2, 512)  # 根据实际输出尺寸调整
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = Net().to(device)
# 使用更小的学习率
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# print(model)

criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=0.001)



def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 创建更美观且功能正常的进度条
    pbar = tqdm(enumerate(train_loader), 
                total=len(train_loader), 
                desc=f"Epoch {epoch:2d}", 
                ncols=100,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 实时更新进度条信息
        if batch_idx % 10 == 0:  # 更频繁地更新信息
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # 更新进度条
        pbar.update()
    
    pbar.close()
    
    train_acc = 100. * correct / total
    train_loss = running_loss / len(train_loader)
    
    # 在进度条结束后打印详细信息
    print(f'\nTrain Epoch: {epoch}  Average loss: {train_loss:.4f}  '
          f'Accuracy: {correct}/{total} ({train_acc:.2f}%)')
    return train_loss, train_acc

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    
    # 为测试也添加进度条
    pbar = tqdm(enumerate(test_loader), 
                total=len(test_loader),
                desc=f"Testing  ", 
                ncols=100,
                leave=False,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    with torch.no_grad():
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 更新进度条
            pbar.update()
    
    pbar.close()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    print(f'Test  Epoch: {epoch}  Average loss: {test_loss:.4f}  '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)')
    return test_loss, test_acc
# 训练模型
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

actual_epochs = 0

for epoch in range(1, 21):  # 增加训练轮数
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    actual_epochs += 1
    
    # 添加早停机制
    if epoch > 5 and test_acc > 70:  # 如果测试准确率超过70%则停止
        print(f"达到目标准确率 {test_acc:.2f}%，提前停止训练")
        break

plt.figure(figsize=(12, 4))
epochs_range = range(1, actual_epochs + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 显示测试图像和预测结果
test_data_iter = iter(test_loader)
test_images, test_labels = next(test_data_iter)

# 显示第一张测试图像，需要调整通道顺序从(C,H,W)到(H,W,C)
plt.imshow(test_images[0].permute(1, 2, 0))
plt.show()

# 预测第一张图像
model.eval()
with torch.no_grad():
    test_image = test_images[0].unsqueeze(0).to(device)
    output = model(test_image)
    prediction = torch.softmax(output, dim=1)
    print(prediction.cpu().numpy()[0])