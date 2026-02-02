from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PyraMLP
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math
from pynvml import *  # Import for NVIDIA GPU monitoring

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.train_epoch_times = []  # Store training time per epoch
        self.inference_times = []    # Store inference time per batch
        # Initialize NVML for GPU monitoring
        if self.args.use_gpu and torch.cuda.is_available():
            nvmlInit()

    def _build_model(self):
        model_dict = {
            'PyraMLP': PyraMLP,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        return mse_criterion, mae_criterion

    def _get_model_params(self):
        """Calculate model parameters in K"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params / 1000  # Convert to K

    def _get_gpu_memory(self):
        """Get GPU memory usage in MiB for current process"""
        if self.args.use_gpu and torch.cuda.is_available():
            try:
                current_pid = os.getpid()
                device_index = torch.cuda.current_device()
                handle = nvmlDeviceGetHandleByIndex(device_index)
                processes = nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in processes:
                    if proc.pid == current_pid:
                        memory_used = proc.usedGpuMemory / 1024 / 1024
                        return memory_used
                print(f"No GPU memory usage found for PID: {current_pid}")
                return 0.0
            except NVMLError as e:
                print(f"Failed to get GPU memory: {e}")
                return 0.0
            except Exception as e:
                print(f"Other error: {e}")
                return 0.0
        return 0.0

    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                pred = outputs
                true = batch_y

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        mse_criterion, mae_criterion = self._select_criterion()

        # Print model parameters
        param_count = self._get_model_params()
        print(f"Model Parameters: {param_count:.2f}K")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_start_time = time.time()

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)



                loss = mae_criterion(outputs, batch_y)
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

            epoch_time = time.time() - epoch_start_time
            self.train_epoch_times.append(epoch_time)

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, mae_criterion, is_test=False)
            test_loss = self.vali(test_data, test_loader, mse_criterion)

            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)

        # Calculate and print training stats
        avg_train_time = np.mean(self.train_epoch_times) if self.train_epoch_times else 0
        gpu_memory = self._get_gpu_memory()
        print(f"Average Training Time per Epoch: {avg_train_time:.4f}s")
        print(f"GPU Memory Usage: {gpu_memory:.2f}MiB")

        # Cleanup NVML
        if self.args.use_gpu and torch.cuda.is_available():
            nvmlShutdown()

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        self.inference_times = []  # Reset inference times
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            print(
                f"DEBUG: Dataset length: {len(test_data)}, Loader length: {len(test_loader)}, Batch Size: {self.args.batch_size}")
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_start_time = time.time()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                inference_time = time.time() - batch_start_time
                self.inference_times.append(inference_time)

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # preds = np.array(preds)
        # trues = np.array(trues)
        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # Calculate metrics and stats
        mae, mse = metric(preds, trues)
        param_count = self._get_model_params()
        avg_train_time = np.mean(self.train_epoch_times) if self.train_epoch_times else 0
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        gpu_memory = self._get_gpu_memory()

        # Print results
        print(f'mse:{mse:.4f}, mae:{mae:.4f}')
        print(f'Model Parameters: {param_count:.2f}K')
        print(f'Average Training Time per Epoch: {avg_train_time:.4f}s')
        print(f'Average Inference Time per Batch: {avg_inference_time:.4f}s')
        print(f'GPU Memory Usage: {gpu_memory:.2f}MiB')

        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write(f'mse:{mse:.4f}, mae:{mae:.4f}\n')
            f.write(f'Model Parameters: {param_count:.2f}K\n')
            f.write(f'Average Training Time per Epoch: {avg_train_time:.4f}s\n')
            f.write(f'Average Inference Time per Batch: {avg_inference_time:.4f}s\n')
            f.write(f'GPU Memory Usage: {gpu_memory:.2f}MiB\n')
            f.write('\n')

        # Save .npy files
        np.save(folder_path + 'metrics.npy', np.array([mae, mse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # Cleanup NVML
        if self.args.use_gpu and torch.cuda.is_available():
            nvmlShutdown()

        return