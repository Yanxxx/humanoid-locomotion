import torch

# 替换成你的 .pt 文件路径
model_path = 'policy-2999.pt'

try:
    # 加载 TorchScript 模型
    model = torch.jit.load(model_path)

    # 1. 打印模型的网络结构
    print("--- 1. Model Architecture ---")
    print(model)

    # 2. 打印模型的前向传播代码（如果可用）
    # 这对于理解模型逻辑非常有帮助
    print("\n--- 2. Model Forward Pass Code ---")
    print(model.code)

except RuntimeError as e:
    # 如果文件损坏或不是一个有效的TorchScript文件，会在这里报错
    print(f"Error loading model: {e}")
