import paddle.nn as nn
from ppim import deit_b_distilled_384
import paddle
import os

deit = deit_b_distilled_384(pretrained=False)
DROP_RATIO = 0.3

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


def get_model(param_path="./deit.pdparams"):
    x = paddle.rand([1, 3, 384, 384])
    # 实例化模型
    model = ViT(deit)
    # 导入上一轮的参数
    if os.path.exists(param_path):
        state_dict = paddle.load(param_path)
        model.load_dict(state_dict)
        print("load params")
    y = model(x)
    print(y.shape)
    return model
