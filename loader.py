from paddle.io import DataLoader
from paddle.vision.datasets import Cifar100
import paddle.vision.transforms as T
import numpy as np

# 训练数据增强
train_transforms = T.Compose([
    T.Resize(384, interpolation='bicubic'),
    T.CenterCrop(384),
    # 随机水平翻转
    T.RandomHorizontalFlip(0.5),
    # 随机旋转
    T.RandomRotation(15),
    # 因为数据集是以pillow的格式读入，因此需要转为tensor
    T.ToTensor(),
    # 归一化
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # 因为paddle目前的版本，GPU环境下Dataloader无法处理tensor，因此需要转为ndarray
    np.array
])
# 验证集数据增强
val_transforms = T.Compose([
    T.Resize(384, interpolation='bicubic'),
    T.CenterCrop(384),
    # 因为数据集是以pillow的格式读入，因此需要转为tensor
    T.ToTensor(),
    # 归一化
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # 因为paddle目前的版本，GPU环境下Dataloader无法处理tensor，因此需要转为ndarray
    np.array
])


def get_train_loader(BATCH_SIZE=24):
    train_dataset = Cifar100(mode="train", transform=train_transforms, backend="pil")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader


def get_val_loader(BATCH_SIZE=24):
    val_dataset = Cifar100(mode="test", transform=val_transforms, backend="pil")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return val_dataloader
