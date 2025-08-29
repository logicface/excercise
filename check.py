import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}",1)
    print(f"GPU数量: {torch.cuda.device_count()}",2)
    print(f"当前GPU: {torch.cuda.get_device_name(0)}",3)

import torch
print(torch.__file__)  # 查看PyTorch安装路径
"check.py"