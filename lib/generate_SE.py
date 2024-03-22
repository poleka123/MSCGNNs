import numpy as np
import networkx as nx
from node2vec import Node2Vec


def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())

    return G

adj_mx = np.load('../data_set/SmallScaleAggregation/adj.npy')
graph = nx.from_numpy_matrix(adj_mx)
# Precompute probabilities and generate walks
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=1)
# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)
model.wv.save_word2vec_format('../data_set/SmallScaleAggregation/SE(SmallScaleAggregation).txt')