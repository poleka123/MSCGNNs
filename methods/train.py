# encoding utf-8
import torch

from utils.utils import Un_Z_Score


def Train(model, optimizer, loss_meathod, NATree, data_set, batch_size, device):
    permutation = torch.randperm(data_set['train_input'].shape[0])
    epoch_training_losses = []
    loss_mean = 0.0
    for i in range(0, data_set['train_input'].shape[0], batch_size):
        model.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = data_set['train_input'][indices], data_set['train_target'][indices]
        X_timestamp, y_timestamp = data_set['train_input_time'][indices], data_set['train_target_time'][indices]
        ids = data_set['ids']
        all_Kmask = data_set['all_Kmask']

        # if torch.cuda.is_available():
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        X_timestamp = X_timestamp.to(device)
        y_timestamp = y_timestamp.to(device)
        std = torch.tensor(data_set['data_std']).to(device)
        mean = torch.tensor(data_set['data_mean']).to(device)
        # else:
        #     std = torch.tensor(data_set['data_std'])
        #     mean = torch.tensor(data_set['data_mean'])
        perd = model(X_batch, NATree, X_timestamp, ids, all_Kmask)
        perd, y_batch = Un_Z_Score(perd, mean, std), Un_Z_Score(y_batch, mean, std)
        loss = loss_meathod(perd, y_batch)

        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        loss_mean = sum(epoch_training_losses)/len(epoch_training_losses)
        if i % 50 == 0:
            print("Loss Mean: " + str(loss_mean))
    return loss_mean