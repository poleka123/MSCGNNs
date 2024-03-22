# encoding utf-8
import numpy as np
import pandas as pd
from utils.utils import *
from torch.utils.data import DataLoader

def Data_load(args):
    filename = args.filename
    timesteps_input = args.timesteps_input
    timesteps_output = args.timesteps_output
    file = files[filename]
    filepath = "./data_set/"
    # metrla
    # data = np.load(filepath + file[0])['data'][:5760].astype(np.float32)
    # pems08
    data = np.load(filepath + file[0])['data'][:8640].astype(np.float32)
    # pems04
    # data = np.load(filepath + file[0])['data'][:2304].astype(np.float32)

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    # 15 day
    # data = np.load(filepath + file[0])['data'][:4320].astype(np.float32)
    # small
    # data = pd.read_csv("./data_set/SmallScaleAggregation/V_flow_50.csv", header=None).head(8640).to_numpy(np.float32)
    # data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    data_cp = data
    input_features = data_cp.shape[2]
    num_nodes = data_cp.shape[1]
    data, data_mean, data_std = Z_Score(data)
    data_length = data.shape[0]
    index_1 = int(data_length * 0.8)
    index_2 = int(data_length * 0.9)
    train_original_data = data[:index_1]
    val_original_data = data[index_1: index_2]
    test_original_data = data[index_2:]

    train_input, train_target = generate_dataset(train_original_data, timesteps_input, timesteps_output)
    evaluate_input, evaluate_target = generate_dataset(val_original_data, timesteps_input, timesteps_output)
    test_input, test_target = generate_dataset(test_original_data, timesteps_input, timesteps_output)

    data_set = {}
    data_set['train_input'], data_set['train_target'], data_set['eval_input'], data_set['eval_target'], \
    data_set['test_input'], data_set['test_target'], data_set['data_mean'], data_set['data_std'],\
    data_set['input_features'], data_set['num_nodes'] = train_input, train_target, evaluate_input, evaluate_target, test_input, test_target,\
                                 data_mean, data_std, input_features, num_nodes
    return data_set

def Data_load_with_time_embedding(args):
    filename = args.filename
    timesteps_input = args.timesteps_input
    timesteps_output = args.timesteps_output
    file = files[filename]
    filepath = "./data_set/"

    data = np.load(filepath + file[0])['data'][:8640].astype(np.float32)
    # small
    # data = pd.read_csv("./data_set/SmallScaleAggregation/V_flow_50.csv", header=None).head(8640).to_numpy(np.float32)
    # data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    data_cp = data
    input_features = data_cp.shape[2]
    num_nodes = data_cp.shape[1]
    data, data_mean, data_std = Z_Score(data)
    data_length = data.shape[0]

    data_flow = np.expand_dims(data[...,0], axis=-1)
    feature_list = [data_flow]
    # numerical time_in_day
    steps_per_day = 288
    time_ind = [i%steps_per_day/steps_per_day for i in range(data.shape[0])]
    time_ind = np.array(time_ind)
    time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
    feature_list.append(time_in_day)

    # numerical day_in_week
    day_in_week = [(i // steps_per_day)%7 for i in range(data.shape[0])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, num_nodes, 1]).transpose((2, 1, 0))
    feature_list.append(day_in_week)

    # data_fusion = np.concatenate([x for x in feature_list], axis=-1).astype(np.float32)
    data_fusion = np.concatenate([x for x in feature_list], axis=-1)


    index_1 = int(data_length * 0.8)
    index_2 = int(data_length * 0.9)
    train_original_data = data_fusion[:index_1]
    val_original_data = data_fusion[index_1: index_2]
    test_original_data = data_fusion[index_2:]

    train_input, train_target = generate_dataset(train_original_data, timesteps_input, timesteps_output)
    evaluate_input, evaluate_target = generate_dataset(val_original_data, timesteps_input, timesteps_output)
    test_input, test_target = generate_dataset(test_original_data, timesteps_input, timesteps_output)

    data_set = {}
    data_set['train_input'], data_set['train_target'], data_set['eval_input'], data_set['eval_target'], \
    data_set['test_input'], data_set['test_target'], data_set['data_mean'], data_set['data_std'],\
    data_set['input_features'], data_set['num_nodes'] = train_input, train_target, evaluate_input, evaluate_target, test_input, test_target,\
                                 data_mean, data_std, input_features, num_nodes
    return data_set