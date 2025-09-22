import numpy as np
import torch
import matplotlib.pyplot as plt

import math

import torch
import torch.nn.functional as F
import os
import pandas as pd
import collections
from itertools import repeat


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def one_hot(label,num_classes):
    '''
        label:[n,1,d,w] --->[n,num_classes,d,w]
    '''
    label = label.long()  #[n,d,w]
    label = F.one_hot(label, num_classes)
    label =  torch.transpose(torch.transpose(label,1,3),2,3)
    return label


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, min_epochs=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.min_epochs = min_epochs

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if (self.counter >= self.patience) and self.counter >= self.min_epochs:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def _ntuple(n):
    """A `to_tuple` function generator.

    It returns a function, this function will repeat the input to a tuple of
    length ``n`` if the input is not an Iterable object, otherwise, return the
    input directly.

    Args:
        n (int): The number of the target length.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=0):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    print(pos_embed, src_shape, dst_shape)
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = F.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


class Myreport:
    def __init__(self, confusion_metrix=None):
        self.__confusion = confusion_metrix

    def __statistics_confusion(self,y_true,y_predict):
        self.__confusion = np.zeros((5, 5))
        for i in range(y_true.shape[0]):
            self.__confusion[y_predict[i]][y_true[i]] += 1

    def __cal_Acc(self):
        return np.sum(self.__confusion.diagonal()) / np.sum(self.__confusion)

    def __cal_Pc(self):
        return self.__confusion.diagonal() / np.sum(self.__confusion, axis=1)

    def __cal_Rc(self):
        return self.__confusion.diagonal() / np.sum(self.__confusion, axis=0)

    def __cal_F1score(self,PC,RC):
        return 2 * np.multiply(PC, RC) / (PC + RC)
    

    def __cal_PA(self):
        return np.sum(self.__confusion.diagonal()) / np.sum(self.__confusion)

    def __cal_mPA(self):
        return np.nanmean(self.__confusion.diagonal() / np.sum(self.__confusion, axis=1))

    def __cal_mIoU(self):
        return np.nanmean(self.__confusion.diagonal() / (np.sum(self.__confusion, axis=1) + np.sum(self.__confusion, axis=0) - self.__confusion.diagonal()))

    def report(self, classNames=None):
        # Acc = self.__cal_Acc()
        # Pc = self.__cal_Pc()
        # Rc = self.__cal_Rc()
        # F1score = self.__cal_F1score(Pc,Rc)

        pa = self.__cal_PA()
        mPA = self.__cal_mPA()
        mIoU = self.__cal_mIoU()

        # str = "Class Name\t\tpa\t\tmPA\t\tmIoU\n"
        # for i in range(len(classNames)):
        #    str += f"{classNames[i]}   \t\t\t{format(Pc[i],'.2f')}   \t\t\t{format(Rc[i],'.2f')}" \
        #           f"   \t\t\t{format(F1score[i],'.2f')}\n"
        # str += f"accuracy is {format(Acc,'.2f')}"
        return pa, mPA, mIoU


def Get_DataPath(Data_Path, Data_Range):
    MOD_Day_CPP_Path  = Data_Path['MOD_Day_CPP_Path'].dropna()
    MOD_Day_SEVI_Path = Data_Path['MOD_Day_SEVI_Path'].dropna()
    MYD_Day_CPP_Path  = Data_Path['MYD_Day_CPP_Path'].dropna()
    MYD_Day_SEVI_Path = Data_Path['MYD_Day_SEVI_Path'].dropna()

    MOD_Night_CPP_Path  = Data_Path['MOD_Night_CPP_Path'].dropna()
    MOD_Night_SEVI_Path = Data_Path['MOD_Night_SEVI_Path'].dropna()
    MYD_Night_CPP_Path  = Data_Path['MYD_Night_CPP_Path'].dropna()
    MYD_Night_SEVI_Path = Data_Path['MYD_Night_SEVI_Path'].dropna()

    CPP_Path_All = pd.concat([MOD_Day_CPP_Path, MYD_Day_CPP_Path, MOD_Night_CPP_Path, MYD_Night_CPP_Path], ignore_index=True)
    SEVI_Path_All = pd.concat([MOD_Day_SEVI_Path, MYD_Day_SEVI_Path, MOD_Night_SEVI_Path, MYD_Night_SEVI_Path], ignore_index=True)

    CPP_Path = []
    SEVI_Path = []
    for i in range(len(CPP_Path_All)):
        if CPP_Path_All[i].split('CPP')[-1] == SEVI_Path_All[i].split('SEVI')[-1]:
            sub_Lat = float(CPP_Path_All[i].split('CPP')[-1].split('_')[-2])
            sub_Lon = float(CPP_Path_All[i].split('CPP')[-1].split('_')[-3])
            if sub_Lat < -Data_Range or sub_Lat > Data_Range or sub_Lon < -Data_Range or sub_Lon > Data_Range:
                continue
            CPP_Path.append(CPP_Path_All[i])
            SEVI_Path.append(SEVI_Path_All[i])
    # print(f'CPP_num: {len(CPP_Path)}, SEVI_num: {len(SEVI_Path)}')
    return CPP_Path, SEVI_Path