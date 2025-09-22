import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
import warnings
import numpy as np
import json
import random

from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim



from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
warnings.filterwarnings('ignore')

from timm.scheduler.cosine_lr import CosineLRScheduler


class Exp_SR(Exp_Basic):
    def __init__(self, args):
        super(Exp_SR, self).__init__(args)

    def _build_model(self):

        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
        return model_optim

    def _select_criterion(self, loss_name='MAE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'CrossEntrophy':
            return nn.CrossEntropyLoss(ignore_index=255)
            # return Custom_CrossEntropy
        elif loss_name == 'MAE':
            return nn.L1Loss()
            

    def train(self, setting):

        time_now = time.time()
        
        # dataloader
        train_set, train_loader = self._get_data(flag='train')
        vali_set, vali_loader = self._get_data(flag='val')

        train_steps = len(train_loader)
        start_epoch = 0
        
        # checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        # log
        log_writer = SummaryWriter(path)

        # early stopping
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        # optimizer
        model_optim = self._select_optimizer()
        
        # lr scheduler
        # scheduler = ReduceLROnPlateau(model_optim, mode='min', factor=0.2, patience=2, verbose=True)
        scheduler = CosineLRScheduler(
                                        model_optim,
                                        t_initial=200,
                                        lr_min=(self.args.learning_rate * 1e-1),
                                        warmup_t=15,            # warmup epoch
                                        warmup_lr_init=(self.args.learning_rate * 1e-2),
                                        cycle_limit=1
                                    )
        # loss func
        # criterion_cls = self._select_criterion(self.args.loss_classification)
        criterion_reg = self._select_criterion(self.args.loss_regression)
        criterion = criterion_reg
        
        # resume
        if self.args.use_checkpoint:
            start_epoch = self.load_checkpoint(path, self.model, model_optim, scheduler, early_stopping)
            print(f'All state reloaded, resume train process from epoch {start_epoch}')


        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (X, y) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                    
                X = torch.tensor(X, dtype=torch.float32).to(self.device) 
                y = torch.tensor(y, dtype=torch.float32).to(self.device)
                
                out = self.model(X)

                loss = criterion(out, y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

                if i % 1000 == 0:
                    log_writer.add_image('Train/loss', loss.item(), epoch * train_steps + i)

            # log
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            log_writer.add_scalar('Loss/train', train_loss, epoch + 1)
            log_writer.add_scalar('Loss/Vali', vali_loss, epoch + 1)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Valid".format(
                    epoch + 1, train_steps, train_loss, vali_loss))


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            self.save_checkpoint(epoch + 1, self.model, model_optim, scheduler, path, vali_loss)
    
        return self.model

    def vali(self, vali_loader, criterion):

        st = time.time()
        self.model.eval()
        with torch.no_grad():
            valid_loss = []

            for (X, y) in enumerate(vali_loader):

                X = torch.tensor(X, dtype=torch.float32).to(self.device) 
                y = torch.tensor(y, dtype=torch.float32).to(self.device)
                
                out = self.model(X)

                loss = criterion(out, y)

                valid_loss.append(loss.item())

            valid_loss = np.average(valid_loss)

        print(f'validation using time:', time.time()-st)
            
        self.model.train()
        return valid_loss
    
    def test(self, setting, test=0, flag='val'):
        test_set, test_loader = self._get_data(flag)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting + '/checkpoint.pth'), map_location=torch.device(f'cuda:{self.args.gpu}')))

        folder_path = os.path.join(self.args.checkpoints, setting + '/results')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        test_metrics = {f'{self.args.task_name}': {'mae':[], 'rmse':[], 'psnr':[], 'ssim':[]}}
        st = time.time()

        self.model.eval()
        with torch.no_grad():

            for (X, y) in enumerate(test_loader):

                X = torch.tensor(X, dtype=torch.float32).to(self.device)
                y = torch.tensor(y, dtype=torch.float32).to(self.device)

                out = self.model(X)
                label = y.detach().cpu().numpy()
                pred = out.detach().cpu().numpy()

                test_metrics[self.args.task_name]['rmse'].append(np.sqrt(mean_squared_error(label, pred)))
                test_metrics[self.args.task_name]['mae'].append(mean_absolute_error(label, pred))
            
                test_metrics[self.args.task_name]['psnr'].append(compare_psnr(label, pred, data_range=1))
                test_metrics[self.args.task_name]['ssim'].append(compare_ssim(label, pred, data_range=1))

        # result save
        for metric in test_metrics[self.args.task_name].keys():
            test_metrics[self.args.task_name][metric] = float((np.mean(test_metrics[self.args.task_name][metric])))

        with open(os.path.join(folder_path, f'{flag}_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)

        print(f'using time:', time.time()-st)
        return

   
    def save_checkpoint(self, epoch: int,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler._LRScheduler,
                        path: str, val_loss):
                                   
            checkpoint = {
                "epoch": epoch,                    
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss,
                "rng_states": {                    
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state()
                }
            }
            torch.save(checkpoint, f"{path}/latest_checkpoint.pth")

    def load_checkpoint(self, path: str,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler._LRScheduler, early_stopping):
                    
                        
        checkpoint = torch.load(f'{path}/latest_checkpoint.pth', map_location=torch.device(f'cuda:{self.args.gpu}'), weights_only=False)
        
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        early_stopping.best_score = -checkpoint["val_loss"]
        early_stopping.val_loss_min = checkpoint["val_loss"]

        if scheduler and checkpoint["scheduler"]:
            scheduler.load_state_dict(checkpoint["scheduler"])
        
        torch.set_rng_state(checkpoint["rng_states"]["torch"].cpu())
        random.setstate(checkpoint["rng_states"]["python"])
        np.random.set_state(checkpoint["rng_states"]["numpy"])
        
        return checkpoint["epoch"]


    

