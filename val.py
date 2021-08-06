from loader import get_val_loader
from layer import get_model
import paddle

val_dataloader = get_val_loader()
model = get_model()
ce_loss = paddle.nn.CrossEntropyLoss()


def val():
    accuracy = paddle.metric.Accuracy()
    model.eval()
    with paddle.no_grad():
        summary = []
        for j, (eval_data, eval_label) in enumerate(val_dataloader()):
            eval_label_hat = model(eval_data)
            eval_loss = ce_loss(eval_label_hat, eval_label)
            correct = accuracy.compute(eval_label_hat, eval_label)
            accuracy.update(correct)
            acc = accuracy.accumulate()
            summary.append(acc)
            accuracy.reset()

    print("[eval]loss:%f,acc:%f" % (eval_loss, sum(summary) / len(summary)))

if __name__=="__main__":
    val()
