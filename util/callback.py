import torch
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, model, path, mode='min', compare=None, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0.0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        self.model = model
        self.path = path
        self.mode = mode
        if compare is None:
            if mode == 'min':
                self.compare = lambda a, b: a < b
            elif mode == 'max':
                self.compare = lambda a, b: a > b
            else:
                assert 0
        else:
            self.compare = compare

    def __call__(self, score):
        score = float(score)
        if self.best_score == 0 or self.compare(score, self.best_score + self.delta):
            if self.verbose:
                print(" {:.6} improved to {:.6}".format(self.best_score, score))
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop, self.best_score


class LearningSchedual(object):
    def __init__(self, optimizer, epochs, end_epoch, train_steps, lr):
        self.optimizer = optimizer
        self.train_steps = train_steps
        self.epochs = epochs
        self.end_epoch = end_epoch
        self.lr = lr

        self.warm_steps = 1
        self.all_steps_without_warm = self.end_epoch * train_steps - self.warm_steps
        string = 'init '
        for i, key in enumerate(lr.keys()):
            self.optimizer.param_groups[i]['lr'] = self.lr[key] * 1 / self.warm_steps
            string += '{} lr:{}, '.format(key, self.optimizer.param_groups[i]['lr'])
        print(string)

    def update_lr(self, epoch, step):
        global_step = epoch * self.train_steps + step + 1
        global_step_without_warm_step = epoch * self.train_steps + step + 1 - self.warm_steps
        if epoch + 1 >= self.end_epoch:
            pass
        elif global_step < self.warm_steps:
            for i, key in enumerate(self.lr.keys()):
                self.optimizer.param_groups[i]['lr'] = self.lr[key] * global_step / self.warm_steps
        elif global_step == self.warm_steps:
            for i, key in enumerate(self.lr.keys()):
                self.optimizer.param_groups[i]['lr'] = self.lr[key]
        elif step == 0:
            rate = (1 - global_step_without_warm_step / self.all_steps_without_warm)
            for i, key in enumerate(self.lr.keys()):
                self.optimizer.param_groups[i]['lr'] = self.lr[key] * rate
        lr = {}
        for i, key in enumerate(self.lr.keys()):
            lr[key] = self.optimizer.param_groups[i]['lr']
        return lr


class FGM(object):
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def re_init_bert(bert_model, layer_list):
    """
    :param bert_model:
    :param layer_list: [12,11,10,9,8,7]
    :return:
    """
    for name, param in bert_model.named_parameters():
        for i in layer_list:
            if str(i) in name:
                param.data = None
