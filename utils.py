import matplotlib.pyplot as plt


def show_imgs(imgs):
    #imgs是一个列表，列表里是多个tensor对象
    #定义总的方框的大小
    plt.figure(figsize=(3*len(imgs),3), dpi=80)
    for i in range(len(imgs)):
        #定义小方框
        plt.subplot(1, len(imgs), i + 1)
        #matplotlib库只能识别numpy类型的数据，tensor无法识别
        imgs[i]=imgs[i].transpose([1,2,0]).numpy()
        #展示取出的数据
        plt.imshow(imgs[i])
        #设置坐标轴
        plt.xticks([])
        plt.yticks([])