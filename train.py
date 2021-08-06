from loader import get_train_loader, get_val_loader
from layer import get_model
import paddle
import paddle.nn.functional as F

train_dataloader = get_train_loader()
val_dataloader = get_val_loader()
model = get_model()


def train(learning_rate=1e-4, epoch_num=100, alpha=5):
    '''
    开局使用AdamW加速收敛
    当准确率达到一定程度后切换Momentum调优
    '''
    optimizer = paddle.optimizer.AdamW(learning_rate=learning_rate, parameters=model.parameters())
    '''
    scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=learning_rate,decay_steps=20,end_lr=learning_rate/10)
    optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,parameters=model.parameters(),weight_decay=1e-2)
    '''

    # 交叉熵损失
    ce_loss = paddle.nn.CrossEntropyLoss()
    accuracy = paddle.metric.Accuracy()
    # 最优准确率
    max_score = 0.
    # 最优准确率对应轮数
    ex_epoch = 0
    for epoch in range(epoch_num):
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
            loss = CE_loss + alpha * KL_loss
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

if __name__=="__main__":
    train()
