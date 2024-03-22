import math
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.data_load import Data_load
from utils.utils import *
from methods.train import Train
from methods.evaluate import Evaluate
from torch.utils.data import DataLoader
import logger

from model.sodeicnet import SODEGCN
import numpy as np

torch.cuda.current_device()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='pems08')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (default: 0.05)')
parser.add_argument('--ksize', type=int, default=3, help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=2, help='# of levels (default: 1)')
parser.add_argument('--timesteps_input', type=int, default=12)
parser.add_argument('--timesteps_output', type=int, default=12)
parser.add_argument('--nhid', type=int, default=16, help='number of hidden units per layer (default: 32)')
parser.add_argument('--spatial_channels', type=int, default=8, help='number of gcn per layer (default: 8)')
parser.add_argument('--time_slice', type=int, default=12)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--model_name', type=str, default='ablation_model')
# 生成邻接矩阵
# parser.add_argument('--filename', type=str, default='smallscaleaggregation')
parser.add_argument('--sigma', type=float, default=10, help='sigma for the spatial matrix')
parser.add_argument('--thres', type=float, default=0.5, help='the threshold for the spatial matrix')
parser.add_argument('--num_of_split', type=float, default=4, help='local timesteps')

args = parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(7)
    elogger = logger.Logger('run_log_mscgnn'+args.filename)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_set = Data_load(args)
    sp_matrix = generate_adjmatrix(args)
    # sp_matrix = generate_adjmatrix_with_ids(args)
    # sp_matrix = load_matrix('./data_set/SmallScaleAggregation/adjmat_50.csv')

    # generate data_loader
    train_setting = [data_set['train_input'], data_set['train_target']]
    train_dataset = DatasetPEMS(train_setting)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    eval_setting = [data_set['eval_input'], data_set['eval_target']]
    eval_dataset = DatasetPEMS(eval_setting)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    test_setting = [data_set['test_input'], data_set['test_target']]
    test_dataset = DatasetPEMS(test_setting)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    num_of_nodes = data_set['num_nodes']
    input_features = data_set['input_features']
    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize

    # 归一化邻接矩阵
    A_sp_wave = torch.from_numpy(get_normalized_adj(sp_matrix)).to(device)

    model = SODEGCN(
        num_nodes=num_of_nodes,
        num_features=input_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        A_sp_hat=A_sp_wave,
        hid_channels=args.nhid,
        sp_channels=args.spatial_channels,
        ksize=args.ksize,
        num_levels=args.levels,
        num_of_split=args.num_of_split,
        dropout=args.dropout
    ).to(device)
    # init change lr fucntion
    batches_per_epoch = math.floor(data_set['train_input'].shape[0]/args.batch_size)
    lr_fn = learning_rate_with_decay(args, args.batch_size, batch_denom=args.batch_size,
                                     batches_per_epoch=batches_per_epoch, boundary_epochs=[80,], decay_rates=[1, 0.1])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    L2 = nn.MSELoss()

    for epoch in range(args.epochs):
        print("Train Process")
        permutation = torch.randperm(data_set['train_input'].shape[0])
        epoch_training_losses = []
        loss_mean = 0.0
        is_best_for_now = False
        # train
        for i, [X_batch, y_batch] in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            # change learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_fn(epoch)

            # indices = permutation[i:i+args.batch_size]
            # X_batch, y_batch = data_set['train_input'][indices], data_set['train_target'][indices]

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            std = torch.tensor(data_set['data_std']).to(device)
            mean = torch.tensor(data_set['data_mean']).to(device)
            pred = model(X_batch)
            pred, y_batch = Un_Z_Score(pred, mean, std), Un_Z_Score(y_batch, mean, std)
            loss = L2(pred, y_batch)

            # 损失反向传播
            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
        loss_mean = sum(epoch_training_losses)/len(epoch_training_losses)


        # test
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Evalution Process")
            model.eval()
            eval_input = data_set['eval_input']
            eval_target = data_set['eval_target']
            eval_input = eval_input.to(device)
            eval_target = eval_target.to(device)
            std = torch.tensor(data_set['data_std']).to(device)
            mean = torch.tensor(data_set['data_mean']).to(device)

            pred = model(eval_input)
            val_index = {}
            val_index['MAE'] = []
            val_index['RMSE'] = []
            val_index['sMAPE'] = []
            val_loss = []

            for item in range(1, args.time_slice+1):
                pred_index = pred[:, :, item - 1]
                val_target_index = eval_target[:, :, item - 1]
                pred_index, val_target_index = Un_Z_Score(pred_index, mean, std), Un_Z_Score(val_target_index, mean,
                                                                                             std)

                loss = L2(pred_index, val_target_index)
                val_loss.append(loss)

                filePath = f"./results/{args.filename}/run_log_mscgnn/"
                if not os.path.exists(filePath):
                    os.makedirs(filePath)
                if ((epoch + 1) % 50 == 0) & (epoch != 0) & (epoch > 100):
                    np.savetxt(filePath + "/pred_" + str(epoch) + ".csv", pred_index.cpu(), delimiter=',')
                    np.savetxt(filePath + "/true_" + str(epoch) + ".csv", val_target_index.cpu(), delimiter=',')
                mae = MAE(val_target_index, pred_index)
                val_index['MAE'].append(mae)

                rmse = RMSE(val_target_index, pred_index)
                val_index['RMSE'].append(rmse)

                smape = SMAPE(val_target_index, pred_index)
                val_index['sMAPE'].append(smape)

        print("---------------------------------------------------------------------------------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("Training loss: {}".format(loss_mean))
        elogger.log("Epoch:{}".format(epoch))
        elogger.log(f"Training loss: {loss_mean}")
        for i in range(1, args.time_slice+1):
            print("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                  .format(i * 5, val_loss[-((args.time_slice) - i)],
                          val_index['MAE'][-((args.time_slice) - i)],
                          val_index['RMSE'][-((args.time_slice) - i)],
                          val_index['sMAPE'][-((args.time_slice) - i)]))
            elogger.log("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                        .format(i * 5, val_loss[-((args.time_slice) - i)],
                                val_index['MAE'][-((args.time_slice) - i)],
                                val_index['RMSE'][-((args.time_slice) - i)],
                                val_index['sMAPE'][-((args.time_slice) - i)]))
        elogger.log("-----------")
        print("---------------------------------------------------------------------------------------------------")
