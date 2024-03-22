import numpy as np

def get_adj_matrix(distance_df_filename, num_of_vertices, id_filename=None, normalized_k=0.01):
    """

    :param distance_df_filename:
    :param num_of_vertices:
    :param id_filename:
    :param nomalized_k:
    :return:
    A: np.ndarray, adj matrix
    """
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)

        return adj_mx, None
    else:
        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        
        distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distanceA[:] = np.inf
        if id_filename:
            with open(id_filename, 'r') as f:
                # 把节点id映射从0开始的索引值
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().strip('\n'))}

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distanceA[id_dict[i], id_dict[j]] = distance
            return A, distanceA
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distanceA[i, j] = distance
                for i in range(num_of_vertices):
                    distanceA[i, i]=0
                distanceA_ = distanceA[~np.isinf(distanceA)].flatten()
                std = distanceA_.std()
                adj = np.exp(-np.square(distanceA/std))
                adj[adj < normalized_k] = 0
            return adj
num_node = 50
data = '../data_set/SmallScaleAggregation/distance.csv'
A = get_adj_matrix(data, num_node, None, normalized_k=0.1)
np.save("../data_set/SmallScaleAggregation/adj.npy", A)





