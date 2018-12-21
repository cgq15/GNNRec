from __future__ import division
from __future__ import print_function

import numpy as np

class MinibatchIter(object):
    def __init__(self, labels, u_indices, v_indices, batch_size):

        self.u_indices = u_indices
        self.v_indices = v_indices
        self.labels = labels
        self.edges = list(zip(u_indices, v_indices, labels))
        self.edge_num = len(self.edges)
        print (self.edge_num)
        self.batch_num = 0
        self.batch_size = batch_size

    def next_minibatch(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, self.edge_num)
        batch_edge = zip(*self.edges[start_idx : end_idx])
        batch_u = np.array(batch_edge[0])
        batch_v = np.array(batch_edge[1])
        batch_label = np.array(batch_edge[2])
        return len(batch_label), batch_u, batch_v, batch_label

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.edges = np.random.permutation(self.edges)
        self.batch_num = 0


    def end(self):
        return self.batch_num * self.batch_size >= self.edge_num

