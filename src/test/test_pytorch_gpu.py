import torch

# 检查是否有可用的 GPU 设备
if torch.cuda.is_available():
    # 打印可用的 GPU 设备数量
    print(f"{torch.cuda.device_count()} GPUs available.")
    # 输出当前设备的名称和计算能力
    print(f"Current device: {torch.cuda.get_device_name()}, "
          f"Compute capability: {torch.cuda.get_device_capability()}")
    # 将 Tensor 迁移到 GPU 上进行计算
    device = torch.device("cuda")
    x = torch.rand(5, 3).to(device)
    print(f"Tensor on GPU:\n{x}")
else:
    print("No available GPUs.")
