import tensorflow as tf
print(tf.__version__)
# import pytorch as pt
gpus = tf.config.experimental.list_physical_devices('GPU')

from keras import datasets,layers,models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# print(train_images[0])

train_images ,test_images = train_images / 255.0 , test_images / 255.0

plt.figure(figsize=(20,10))
for i in range(20):
        plt.subplot(5,10,i+1)# 创建子图网格中的第(i+1)个子图，网格大小为5行10列
        plt.xticks([])# 移除x轴刻度标记
        plt.yticks([])# 移除y轴刻度标记
        plt.grid(False)# 关闭网格显示
        plt.imshow(train_images[i], cmap=plt.cm.binary)# 显示训练图像，使用二值色彩映射
        plt.xlabel(train_labels[i])# 设置x轴标签为对应的训练标签
plt.show()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
# print(train_images.shape)

# 创建一个用于图像分类的卷积神经网络模型
# 该模型采用Sequential结构，包含卷积层、池化层、展平层和全连接层
# 输入：形状为(28, 28, 1)的灰度图像数据
# 输出：10个类别的概率分布，用于多分类任务
model = models.Sequential([
        # 第一个卷积层：32个3x3的卷积核，ReLU激活函数，输入图像大小为28x28x1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # 第一个最大池化层：2x2池化窗口，用于降采样和特征提取
        layers.MaxPooling2D((2, 2)),
        # 第二个卷积层：64个3x3的卷积核，ReLU激活函数
        layers.Conv2D(64, (3, 3), activation='relu'),
        # 第二个最大池化层：2x2池化窗口
        layers.MaxPooling2D((2, 2)),
        # 第三个卷积层：64个3x3的卷积核，ReLU激活函数
        layers.Conv2D(64, (3, 3), activation='relu'),
        # 展平层：将多维特征图转换为一维特征向量
        layers.Flatten(),
        # 第一个全连接层：64个神经元，ReLU激活函数
        layers.Dense(64, activation='relu'),
        # 输出层：10个神经元，softmax激活函数，对应10个分类类别
        layers.Dense(10, activation='softmax')
])

# print(model.summary())
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('\nTest accuracy:', test_acc)

plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.show()

pre = model.predict(test_images)
print(pre[0])




# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np

# print(torch.__version__)

# def setup_gpu():
#     """
#     配置GPU环境，包括显存管理和设备选择
#     """
# #     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print(f"GPU设备: {torch.cuda.get_device_name(0)}")
#         print(f"GPU数量: {torch.cuda.device_count()}")
#         print(f"CUDA版本: {torch.version.cuda}")
        
#         # 配置显存使用策略
#         # 方式1: 设置显存按需增长（适用于TensorFlow风格）
#         # 注意：PyTorch默认就是按需分配显存，不需要特殊设置
        
#         # 方式2: 如果需要限制显存使用，可以使用以下方法
#         # torch.cuda.set_per_process_memory_fraction(0.8)  # 限制使用80%显存
        
#         # 清空GPU缓存
#         torch.cuda.empty_cache()
#     else:
#         device = torch.device("cpu")
#         print("未检测到GPU，使用CPU进行训练")
    
#     return device

# # 设置GPU环境
# # device = setup_gpu()
# print(f"Using device: {device}")

# # 数据预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
# ])

# # 加载MNIST数据集

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import os
# import urllib.request
# import ssl

# # 回退到使用TensorFlow加载MNIST数据
# from keras import datasets
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import torch

# class TensorDataset(Dataset):
#     def __init__(self, images, labels, transform=None):
#         self.images = images
#         self.labels = labels
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
        
#         # 添加通道维度
#         if len(image.shape) == 2:
#             image = np.expand_dims(image, axis=0)  # 添加通道维度
#         else:
#             image = image.transpose(2, 0, 1)  # 调整维度顺序 HWC -> CHW
        
#         image = torch.from_numpy(image).float()
#         label = torch.from_numpy(np.array(label)).long()
        
#         if self.transform:
#             # 注意：某些transform可能不适用于Tensor
#             pass
            
#         return image, label

# # 使用TensorFlow加载MNIST数据
# print("使用TensorFlow加载MNIST数据集...")
# (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# # 归一化数据
# train_images = train_images.astype(np.float32) / 255.0
# test_images = test_images.astype(np.float32) / 255.0

# # 创建PyTorch数据集
# train_dataset = TensorDataset(train_images, train_labels)
# test_dataset = TensorDataset(test_images, test_labels)

# print("数据集加载完成！")
# print(f"训练集大小: {len(train_dataset)}")
# print(f"测试集大小: {len(test_dataset)}")


# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# # 定义卷积神经网络模型
# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         # 卷积层块
#         self.conv_layers = nn.Sequential(
#             # 第一个卷积层：32个3x3的卷积核，ReLU激活函数
#             nn.Conv2d(1, 32, kernel_size=3, padding=0),
#             nn.ReLU(),
#             # 第一个最大池化层：2x2池化窗口
#             nn.MaxPool2d(2, 2),
#             # 第二个卷积层：64个3x3的卷积核
#             nn.Conv2d(32, 64, kernel_size=3, padding=0),
#             nn.ReLU(),
#             # 第二个最大池化层：2x2池化窗口
#             nn.MaxPool2d(2, 2),
#             # 第三个卷积层：64个3x3的卷积核
#             nn.Conv2d(64, 64, kernel_size=3, padding=0),
#             nn.ReLU()
#         )
        
#         # 全连接层块
#         self.fc_layers = nn.Sequential(
#             # 展平层在forward中处理
#             # 第一个全连接层：64个神经元
#             nn.Linear(64 * 3 * 3, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             # 输出层：10个神经元
#             nn.Linear(64, 10)
#         )
    
#     def forward(self, x):
#         # 前向传播
#         x = self.conv_layers(x)
#         # 展平操作
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x

# # 创建模型实例并移至设备
# model = CNNModel().to(device)
# print(model)

# # 打印模型各层的输入输出形状
# def print_model_shapes():
#     # 创建一个示例输入张量
#     sample_input = torch.randn(1, 1, 28, 28).to(device)
    
#     print("\nModel layer shapes:")
#     print("Input:", sample_input.shape)
    
#     # 逐层打印形状
#     x = sample_input
#     x = model.conv_layers[0](x)  # Conv2D 1
#     print("After Conv2D(32, 3x3):", x.shape)
#     x = model.conv_layers[1](x)  # ReLU
#     print("After ReLU:", x.shape)
#     x = model.conv_layers[2](x)  # MaxPool2D 1
#     print("After MaxPool2D(2x2):", x.shape)
#     x = model.conv_layers[3](x)  # Conv2D 2
#     print("After Conv2D(64, 3x3):", x.shape)
#     x = model.conv_layers[4](x)  # ReLU
#     print("After ReLU:", x.shape)
#     x = model.conv_layers[5](x)  # MaxPool2D 2
#     print("After MaxPool2D(2x2):", x.shape)
#     x = model.conv_layers[6](x)  # Conv2D 3
#     print("After Conv2D(64, 3x3):", x.shape)
#     x = model.conv_layers[7](x)  # ReLU
#     print("After ReLU:", x.shape)
#     x = x.view(x.size(0), -1)    # Flatten
#     print("After Flatten:", x.shape)
#     x = model.fc_layers[0](x)    # Dense 1
#     print("After Dense(64):", x.shape)
#     x = model.fc_layers[1](x)    # ReLU
#     print("After ReLU:", x.shape)
#     x = model.fc_layers[2](x)    # Dropout
#     print("After Dropout:", x.shape)
#     x = model.fc_layers[3](x)    # Output Dense
#     print("Output:", x.shape)

# print_model_shapes()

# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 训练函数
# def train(epoch):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
        
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         _, predicted = output.max(1)
#         total += target.size(0)
#         correct += predicted.eq(target).sum().item()
        
#         if batch_idx % 100 == 0:
#             print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
#                   f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
#     train_acc = 100. * correct / total
#     train_loss = running_loss / len(train_loader)
#     print(f'Train Epoch: {epoch}\tAverage loss: {train_loss:.4f}\t'
#           f'Accuracy: {correct}/{total} ({train_acc:.2f}%)')
#     return train_loss, train_acc

# # 测试函数
# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
    
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
    
#     test_loss /= len(test_loader)
#     test_acc = 100. * correct / len(test_loader.dataset)
    
#     print(f'\nTest set: Average loss: {test_loss:.4f}, '
#           f'Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)\n')
#     return test_loss, test_acc

# # 训练模型
# train_losses = []
# train_accuracies = []
# test_losses = []
# test_accuracies = []

# for epoch in range(1, 6):
#     train_loss, train_acc = train(epoch)
#     test_loss, test_acc = test()
    
#     train_losses.append(train_loss)
#     train_accuracies.append(train_acc)
#     test_losses.append(test_loss)
#     test_accuracies.append(test_acc)

# # 绘制训练过程
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(range(1, 6), train_accuracies, label='Training Accuracy')
# plt.plot(range(1, 6), test_accuracies, label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(range(1, 6), train_losses, label='Training Loss')
# plt.plot(range(1, 6), test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # 显示测试图像和预测结果
# test_data_iter = iter(test_loader)
# test_images, test_labels = next(test_data_iter)

# # 显示第一张测试图像
# plt.imshow(test_images[0].squeeze(), cmap=plt.cm.binary)
# plt.show()

# # 预测第一张图像
# model.eval()
# with torch.no_grad():
#     test_image = test_images[0].unsqueeze(0).to(device)
#     output = model(test_image)
#     prediction = torch.softmax(output, dim=1)
#     print(prediction.cpu().numpy()[0])