import tensorflow as tf

# 列出所有可用的物理设备
physical_devices = tf.config.list_physical_devices('GPU')
print("physical_devices:", physical_devices)

matrix = tf.random.normal((3, 3))
print("Random matrix:")
print(matrix)

# 如果没有可用的物理设备，则 TensorFlow 将默认使用 CPU
if len(physical_devices) == 0:
    print("No available physical devices.")
else:
    # 遍历所有可用的物理设备
    for device in physical_devices:
        # 检查设备是否为 GPU
        if device.device_type == 'GPU':
            # 设置 TensorFlow 使用该 GPU 设备
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Device name: {device.name}, device type: {device.device_type}")
            print("TensorFlow is using GPU.")
            # 在 GPU 上生成随机矩阵
            with tf.device(device.name):
                matrix = tf.random.normal((3, 3))
                print("Random matrix:")
                print(matrix)
            break
    else:
        print("No available GPUs.")
