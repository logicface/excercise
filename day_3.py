import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 设置数据保存路径
data_path = './data'

# 创建数据目录（如果不存在）
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f'创建数据目录: {data_path}')

# 检查Fashion-MNIST数据集文件是否已存在
def check_fashion_mnist_exists():
    # Fashion-MNIST数据集文件通常保存在以下路径
    mnist_dir = os.path.join(data_path, 'FashionMNIST', 'raw')
    
    # 检查必要的数据集文件是否存在
    required_files = [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    ]
    
    # 如果目录不存在，数据集肯定不存在
    if not os.path.exists(mnist_dir):
        return False
    
    # 检查每个必需的文件是否存在
    for file in required_files:
        file_path = os.path.join(mnist_dir, file)
        if not os.path.exists(file_path):
            return False
    
    return True

# 检查数据集是否已存在
dataset_exists = check_fashion_mnist_exists()
if dataset_exists:
    print("Fashion-MNIST数据集已存在于本地，将直接加载...")
else:
    print("Fashion-MNIST数据集不存在于本地，将开始下载...")

# 定义数据预处理转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量并归一化到[0,1]
    transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1,1]
])

# 加载Fashion-MNIST训练数据集（根据本地是否存在决定是否下载）
train_dataset = datasets.FashionMNIST(
    root=data_path,
    train=True,
    download=not dataset_exists,  # 只有当数据集不存在时才下载
    transform=transform
)

# 加载Fashion-MNIST测试数据集（根据本地是否存在决定是否下载）
test_dataset = datasets.FashionMNIST(
    root=data_path,
    train=False,
    download=not dataset_exists,  # 只有当数据集不存在时才下载
    transform=transform
)

if not dataset_exists:
    print(f"Fashion-MNIST数据集下载完成！")
else:
    print(f"Fashion-MNIST数据集加载完成！")

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 数据集类别标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 打印数据集信息
print(f"Fashion-MNIST数据集下载完成！")
print(f"训练集大小: {len(train_dataset)} 样本")
print(f"测试集大小: {len(test_dataset)} 样本")
print(f"类别数: {len(class_names)}")
print(f"类别标签: {class_names}")

# 可视化几个样本
def visualize_samples():
    # 随机选择6个样本
    indices = np.random.randint(0, len(train_dataset), 6)
    
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(indices):
        image, label = train_dataset[idx]
        plt.subplot(2, 3, i + 1)
        # 转换形状 (C, H, W) -> (H, W, C)
        img = image.squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(class_names[label])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 如果需要可视化样本，可以取消下面这行的注释
visualize_samples()
print(f"第一个训练样本形状: {train_dataset[0][0].shape}")
print(f"第一个测试样本形状: {test_dataset[0][0].shape}")





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道改为1，因为Fashion-MNIST是灰度图像
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

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



# 将原有的列表定义修改为不同的名称
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

actual_epochs = 0

for epoch in range(1, 21):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    # 使用不同的列表名来存储结果
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    actual_epochs += 1

# 添加可视化训练过程的代码
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, actual_epochs+1), train_losses, 'b-', label='训练损失')
plt.plot(range(1, actual_epochs+1), test_losses, 'r-', label='测试损失')
plt.title('训练和测试损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, actual_epochs+1), train_accuracies, 'b-', label='训练准确率')
plt.plot(range(1, actual_epochs+1), test_accuracies, 'r-', label='测试准确率')
plt.title('训练和测试准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率 (%)')
plt.legend()

plt.tight_layout()
plt.show()
