import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_log(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s',
                        datefmt='%y-%m-%d %H:%M',
                        filename=log_file,
                        filemode='w')
    # 定义一个Handler打印INFO及以上级别的日志到sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 设置日志打印格式
    formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s',
                                  datefmt='%y-%m-%d %H:%M')
    console.setFormatter(formatter)
    # 将定义好的console日志handler添加到root logger
    logging.getLogger('').addHandler(console)


def get_feature(x, model, layer_idx):
    layers = model.features
    for idx, layer in enumerate(layers):
        x = layer(x)
        if idx == layer_idx:
            return x


TIME = 0


def total_time(x):
    x = [xx + i * TIME for i, xx in enumerate(x)]
    return x


def smooth_curve(y, delta=5):
    y = [sum(y[max(i - delta, 0): min(i + delta, len(y) - 1)]) / len(y[max(i - delta, 0): min(i + delta, len(y) - 1)])
         for i in range(len(y))]
    return y


def plot_metrics(d, metrics, objs=None, tilte=None, file_name=None):
    for metric in metrics:
        fig, ax = plt.subplots()
        for key, val in d.items():
            if objs is None or key in objs:
                ax.plot(val[metric])
        ax.legend()
        ax.set_xlabel('Iterations')
        ax.set_ylabel(metric)
        if tilte is not None:
            ax.set_tilte(tilte)
        if file_name is not None:
            fig.save_fig(file_name)
        fig.show()


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)
