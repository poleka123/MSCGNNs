import os.path
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from dtaidistance import dtw
from fastdtw import fastdtw

files = {
    'pems03': ['PEMS03/PEMS03.npz', 'PEMS03/PEMS03.csv'],
    'pems04': ['PEMS04/PEMS04.npz', 'PEMS04/PEMS04.csv'],
    'pems07': ['PEMS07/PEMS07.npz', 'PEMS07/PEMS07.csv'],
    'pems08': ['PEMS08/PEMS08.npz', 'PEMS08/PEMS08.csv', '170'],
    'pemsbay': ['PEMSBAY/pems_bay.npz', 'PEMSBAY/distance.csv'],
    'pemsD7M': ['PeMSD7M/PeMSD7M.npz', 'PeMSD7M/distance.csv'],
    'pemsD7L': ['PeMSD7L/PeMSD7L.npz', 'PeMSD7L/distance.csv'],
    'metr-la': ['METR-LA/METR-LA.npz', 'METR-LA/distances_la.csv', 'METR-LA/graph_sensor_ids.txt'],
    'randomuniformity': ['RandomUniformity/V_flow_50.npz', 'RandomUniformity/V_flow_50.csv'],
    'smallscaleaggregation': ['SmallScaleAggregation/V_flow_50.npz', 'SmallScaleAggregation/V_flow_50.csv']
}

class DatasetPEMS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, index):
        sample = self.data[0][index]
        label = self.data[1][index]

        return sample, label


def Z_Score(matrix):
    mean, std = np.mean(matrix), np.std(matrix)
    return (matrix - mean) / (std+0.001), mean, std


def Un_Z_Score(matrix, mean, std):
    return (matrix * std) + mean

def RMSE(v, v_):
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MAE(v, v_):
    return torch.mean(torch.abs(v_ - v))

def SMAPE(v, v_):
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    """
    return torch.mean(torch.abs((v_ - v) / ((torch.abs(v) + torch.abs(v_)) / 2 + 1e-5)))

def load_matrix(file_path):
    return pd.read_csv(file_path, header=None).to_numpy(np.float32)

def get_normalized_adj(W_nodes):
    W_nodes = W_nodes + np.diag(np.ones(W_nodes.shape[0], dtype=np.float32))
    D = np.array(np.sum(W_nodes, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    W_nodes = np.multiply(np.multiply(diag.reshape((-1, 1)), W_nodes),
                         diag.reshape((1, -1)))
    return W_nodes

def generate_dtw_adjmatrix(args):
    filename = args.filename
    file = files[filename]
    filepath = "./data_set/"
    data = np.load(filepath + file[0])['data'][:8640].astype(np.float32)
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]
    if not os.path.exists(f'data_set/{filename}_dtw_distance.npy'):
        data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
        data_mean = data_mean.squeeze().T
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(f'data_set/{filename}_dtw_distance.npy', dtw_distance)

    dist_matrix = np.load(f'data_set/{filename}_dtw_distance.npy')

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1
    dtw_matrix = dtw_matrix.astype('float32')
    return dtw_matrix


def generate_adjmatrix_with_ids(args):
    filename = args.filename
    file = files[filename]
    filepath = "./data_set/"
    with open(filepath + file[2]) as f:
        sensor_ids = f.read().strip().split(",")
    num_node = (np.load(filepath + file[0])['data']).shape[1]
    # builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
    dist_matrix = np.zeros((num_node, num_node), dtype=np.float32)
    dist_matrix[:] = np.inf
    distance_df = pd.read_csv(filepath + file[1], dtype={"from": "str", "to": "str"})
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_matrix[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
    #  save distance matrixs
    if not os.path.exists(f'./data_set/{filename}_spatial_distance.npy'):
        np.save(f'./data_set/{filename}_spatial_distance.npy', dist_matrix)
    dist_matrix = np.load(f'data_set/{filename}_spatial_distance.npy')
    # normalization
    std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma
    sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
    sp_matrix[sp_matrix < args.thres] = 0
    sp_matrix = sp_matrix.astype('float32')

    return sp_matrix






def generate_adjmatrix(args):
    filename = args.filename
    file = files[filename]
    filepath = "./data_set/"
    num_node = (np.load(filepath + file[0])['data']).shape[1]
    # use continuous spatial matrix
    if not os.path.exists(f'data_set/{filename}_spatial_distance.npy'):
        with open(filepath + file[1], 'r') as fp:
            dist_matrix = np.zeros((num_node, num_node)) + float('inf')
            file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
            np.save(f'data_set/{filename}_spatial_distance.npy', dist_matrix)

    dist_matrix = np.load(f'data_set/{filename}_spatial_distance.npy')
    # normalization
    std = np.std(dist_matrix[dist_matrix != float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma
    sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
    sp_matrix[sp_matrix < args.thres] = 0
    sp_matrix = sp_matrix.astype('float32')

    return sp_matrix

def learning_rate_with_decay(args, batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    # boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    boundaries = [epoch for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def generate_asist_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features))

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[0] - (num_timesteps_input + num_timesteps_output) + 1)]
    features, target = [], []
    #  data shape is (len, nodes, timesteps, channels)
    for i, j in indices:
        features.append(
            X[i: i + num_timesteps_input, :, :].transpose((1, 0, 2)))
        target.append(X[i + num_timesteps_input: j, :, 0].transpose((1, 0)))
    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target)),


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)



