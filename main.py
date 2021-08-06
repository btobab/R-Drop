#!/usr/bin/env python
# coding: utf-8


from ppim import deit_b_distilled_384
import paddle
import numpy as np
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.vision.datasets import Cifar100
import paddle.vision.transforms as T
import paddle.nn.functional as F
import matplotlib.pyplot as plt
from utils import show_imgs
import os

## 设置超参数

# 批大小
BATCH_SIZE = 24
# dropout的概率
DROP_RATIO = 0.3
# 学习率
LEARNING_RATE = 3e-4
# 训练轮数
EPOCH_NUM = 100
# KL divergence的损失权重
ALPHA = 5
# 上一轮的模型参数保存路径
PARAM_PATH = "./deit.pdparams"

## 数据预处理

# 实例化deit
deit = deit_b_distilled_384(pretrained=False)
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

# 获取数据集
train_dataset = Cifar100(mode="train", transform=train_transforms, backend="pil")
val_dataset = Cifar100(mode="test", transform=val_transforms, backend="pil")
# 封装为读取器
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 检查数据格式
data, label = next(train_dataloader())
print(data)


## 自定义读取器(笔者用小样本试试水)

def reader(dataset, mode="train"):
    if mode == "train":
        # paricipate:候选数据
        parti = np.arange(0, 1000)
    else:
        parti = np.arange(2000, 2100)
    for k in range(len(parti) // BATCH_SIZE):
        # 随机选取数据
        idxs = np.random.choice(parti, BATCH_SIZE)
        data = []
        labels = []
        for i in range(BATCH_SIZE):
            img, label = dataset[idxs[i]]
            data.append(img)
            labels.append(label)
        data, labels = paddle.to_tensor(data), paddle.to_tensor(labels)
        yield data, labels


## 定义模型架构

class ViT(nn.Layer):
    def __init__(self, deit):
        super(ViT, self).__init__()
        self.deit = deit
        self.nets = nn.Sequential(
            self.deit,
            nn.Linear(1000, 512),
            nn.Tanh(),
            nn.Dropout(DROP_RATIO),
            nn.Linear(512, 100),
            nn.Dropout(DROP_RATIO)
        )

    def forward(self, x):
        return self.nets(x)


x = paddle.rand([1, 3, 384, 384])
# 实例化模型
model = ViT(deit)
# 导入上一轮的参数
if os.path.exists(PARAM_PATH):
    state_dict = paddle.load(PARAM_PATH)
    model.load_dict(state_dict)
    print("load params")
y = model(x)
print(y.shape)

'''
开局使用AdamW加速收敛
当准确率达到一定程度后切换Momentum调优
'''
optimizer = paddle.optimizer.AdamW(learning_rate=LEARNING_RATE, parameters=model.parameters())
'''
scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=LEARNING_RATE,decay_steps=20,end_lr=LEARNING_RATE/10)
optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,parameters=model.parameters(),weight_decay=1e-2)
'''

# 交叉熵损失
ce_loss = paddle.nn.CrossEntropyLoss()
accuracy = paddle.metric.Accuracy()
# 最优准确率
max_score = 0.
# 最优准确率对应轮数
ex_epoch = 0
for epoch in range(EPOCH_NUM):
    model.train()
    for i, (data, label) in enumerate(train_dataloader()):
        summary = []
        # 前向传播两次
        label_hat_A = model(data)
        label_hat_B = model(data)
        # cross entropy loss
        CE_loss = ce_loss(label_hat_A, label) + ce_loss(label_hat_B, label)
        # KL divergence loss
        KL_loss = 0.5 * (F.kl_div(F.softmax(label_hat_A, axis=-1), F.softmax(label_hat_B, axis=-1)) + F.kl_div(
            F.softmax(label_hat_B, axis=-1), F.softmax(label_hat_A, axis=-1)))
        # 损失加权求和
        loss = CE_loss + ALPHA * KL_loss
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 清除梯度
        optimizer.clear_gradients()
        if i % 30 == 0:
            print("[train]epoch:%d,i:%d,loss:%f" % (epoch, i, loss))

    model.eval()
    with paddle.no_grad():
        for j, (eval_data, eval_label) in enumerate(val_dataloader()):
            summary = []
            eval_label_hat = model(eval_data)
            eval_indexs = eval_label_hat.argmax(-1)
            eval_loss = ce_loss(eval_label_hat, eval_label)
            correct = accuracy.compute(eval_label_hat, eval_label)
            accuracy.update(correct)
            acc = accuracy.accumulate()
            summary.append(acc)
            accuracy.reset()

    print("[eval]epoch:%d,loss:%f,acc:%f" % (epoch, eval_loss, sum(summary) / len(summary)))
    if sum(summary) / len(summary) >= max_score:
        max_score = sum(summary) / len(summary)
        ex_epoch = epoch
        paddle.save(model.state_dict(), "./deit_.pdparams")
        print("[eval]saved params deit_")
    print("[eval]ex_epoch:%d,best acc:%f" % (ex_epoch, max_score))
