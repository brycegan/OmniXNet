
import os
import torch
import random
import argparse
import numpy as np
from exp.exp_SR import Exp_SR


if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='OMNIXNET')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='super-resolution',
                        help='task name, options:[cer, cot, cth, clp]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status 1: train 2:test 3:predict')
    parser.add_argument('--raytune', type=bool, default=False, help='hyperparameter searching')
    parser.add_argument('--seed', type=int, default=2021, help='rand seed')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='OMNIXNET',
                        help='model name, options: []')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='', help='dataset name')
    parser.add_argument('--data_path', type=str, default='', help='data file folder')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs') # 50
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss_regression', type=str, default='MAE', help='regression loss function, options:[MSE, MAE, MS_SSIM_L1_LOSS, L1_Perception]')
    parser.add_argument('--loss_classification', type=str, default='CrossEntrophy', help='classification loss function')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,2,4,5', help='device ids of multile gpus')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    # model
    parser.add_argument('--patch_size', type=int, default=4, help='patch size')
    parser.add_argument('--img_size', type=int, default=128, help='img size')
    parser.add_argument('--in_chans', type=int, default=7, help='in channels')
    parser.add_argument('--num_classes', type=int, default=3, help='num classes')
    parser.add_argument('--embed_dim', type=int, default=96, help='embed dim')
    parser.add_argument('--feature_channles', type=list, default=[96, 192, 384,768], help='feature channels, options: [[96, 192, 384,768], [64, 128, 256, 512], [48, 96, 192, 384]]')
    parser.add_argument('--global_branch_depths', type=list, default=[2,2,18,2], help='net depth, options: [[2,2,18,2], [2,2,18,2], [2,2,9,2]]')
    parser.add_argument('--local_branch_depths', type=list, default=[3, 3, 27, 3], help='net depth, options: [[3, 3, 27, 3], [2,2,18,2], [2,2,9,2]]')
    parser.add_argument('--num_heads', type=list, default=[3, 6, 12, 24], help='num heads')
    parser.add_argument('--window_size', type=int, default=4, help='embed dim')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='mlp ratio')
    parser.add_argument('--qkv_bias', type=bool, default=True, help='qkv bias')
    parser.add_argument('--qk_scale', type=None, default=None, help='qk scale')
    parser.add_argument('--drop_rate', type=float, default=0.0 , help='drop rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='drop path rate')
    parser.add_argument('--ape', type=bool, default=False, help='ape')
    parser.add_argument('--patch_norm', type=bool, default=True, help='patch norm')
    parser.add_argument('--use_checkpoint', type=bool, default=False, help='use checkpoint')

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_SR

    if args.is_training == 1:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_{}'.format(
                args.des, 
                args.task_name,
                args.model_id,
                args.model,
                args.data, ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

    elif args.is_training == 2:
        ii = 0
        setting = '{}_{}_{}_{}_{}'.format(
            args.des,
            args.task_name,
            args.model_id,
            args.model,
            args.data, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting, test=1, flag='val')
        exp.test(setting, test=1, flag='test')
        torch.cuda.empty_cache()

