from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)

class MinibatchIter(object):
    def __init__(self, u_features, v_features, adj, labels, 
                u_indices, v_indices, batch_size):
        self.adj = adj
        self.u_features = u_features
        self.v_features = v_features
        self.labels = labels
        self.u_indices = u_indices
        self.v_indices = v_indices

        self.u_num, self.v_num = adj.shape
        self.edge_num = len(labels)
        self.u = np.unique(u_indices)
        self.v = np.unique(v_indices)
        self.batch_num = 0
        self.batch_size = batch_size
        self.edges = list(zip(u_indices, v_indices, labels))
        self.deg_u = np.array([x.nnz for x in self.adj])
        b = adj.transpose()
        self.deg_v = np.array([x.nnz for x in b])

    def next_minibatch(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.labels))
        batch_edge = zip(*self.edges[start_idx : end_idx])
        batch_u = np.array(batch_edge[0])
        batch_v = np.array(batch_edge[1])
        batch_label = np.array(batch_edge[2])
        return len(self.labels), batch_u, batch_v, batch_label

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.edges = np.random.permutation(self.edges)
        self.batch_num = 0


    def end(self):
        return self.batch_num * self.batch_size >= self.edge_num

