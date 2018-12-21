from utils import *
import scipy.sparse as sp
from sparse_tensor_utils import *


class Sampler(object):
    """Sampling and Network Constructing"""

    def __init__(self, placeholders, adj, **kwargs):
        allowed_kwargs = {'num_layers', 'input_dim', 'layer_sizes', 'scope'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        self.input_dim = kwargs.get('input_dim', 1)
        self.layer_sizes = kwargs.get('layer_sizes', [1])
        self.scope = kwargs.get('scope', 'test_graph')

        #self.num_layers = len(self.layer_sizes)
        '''
        with tf.variable_scope(self.scope, reuse=True):
            self.w_s = tf.get_variable('sample_weights', [self.input_dim, 2])
        '''
        self.adj = adj
        self.adj_trans = adj.transpose()
        self.item_samples = None
        self.item_sampled_vals = None
        self.user_samples = None
        self.user_sampled_vals = None

        #self.num_data = tf.shape(self.adj)[0]

    def sampling(self, v, type):
        pass
    
    def gen_sample_list(self):
        pass

'''
class SamplerAdapt(Sampler):
    """Parameters of the sampler are adaptive"""

    def sampling(self, v, type):
        all_support = [[]] * (self.num_layers - 1)
        all_p_u = [[]] * (self.num_layers - 1)
        all_x_u = [[]] * self.num_layers

        # sample top-1 layer
        this_x_v = tf.gather(self.x, v)
        this_adj = tf.gather(self.adj, v)
        this_adj_val = tf.gather(self.adj_val, v)
        all_x_u[self.num_layers - 1] = this_x_v

        # top-down sampling from top-2 layer to the input layer
        for i in range(self.num_layers - 1):
            layer = self.num_layers - i - 2

            u_sampled, support, p_u = self.one_layer_sampling(adj=this_adj, adj_val=this_adj_val, x_v=this_x_v,
                                                              output_size=self.layer_sizes[layer])

            this_x_v = tf.gather(self.x, u_sampled)
            this_adj = tf.gather(self.adj, u_sampled)
            this_adj_val = tf.gather(self.adj_val, u_sampled)

            all_x_u[layer] = this_x_v
            all_support[layer] = support
            all_p_u[layer] = p_u

        return all_x_u, all_support, all_p_u

    def one_layer_sampling(self, adj, adj_val, x_v, output_size):
        """layer wise sampling """

        support, u = from_adjlist(adj, adj_val)
        x_u = tf.gather(self.x, u)
        h_v = tf.reduce_sum(tf.matmul(x_v, tf.expand_dims(self.w_s[:, 1], -1)))
        h_u = tf.matmul(x_u, tf.expand_dims(self.w_s[:, 0], -1))
        attention = tf.reshape(1 / tf.cast(output_size, tf.float32) * (tf.nn.relu(h_v + h_u) + 1), [-1])
        g_u = tf.reshape(tf.nn.relu(h_u) + 1, [-1])

        p1 = tf.cast(tf.sqrt(tf.sparse_reduce_sum(tf.square(support), axis=0)), dtype=tf.float32) * attention * g_u
        p1 = p1 / tf.reduce_sum(p1)

        samples = tf.cast(tf.multinomial([tf.log(p1)], output_size), tf.int64)

        u_sampled = tf.gather(u, samples[0])
        p_u = tf.gather(p1, samples[0])
        support = SparsePlus(support)
        support_sampled = support.gather_columns(samples[0])

        return u_sampled, support_sampled, p_u
'''
class SimpleSampler(Sampler):

    def sampling(self, v, type):
        if type == 'item':
            return self.item_samples[v], self.item_sampled_vals[v]
        elif type == 'user':
            return self.user_samples[v], self.user_sampled_vals[v]
    
    def gen_sample_list(self):
        samples = []
        sampled_vals = []
        for node in range(self.adj.shape[0]):
            neib = self.adj[node]
            if neib.nnz >= self.layer_sizes:
                sample = np.random.choice(neib.indices, self.layer_sizes, replace=False)
                
            else:
                sample = np.random.choice(neib.indices, self.layer_sizes, replace=True)
            val = neib[0, sample].toarray()
            samples.append(sample)
            sampled_vals.append(val)
        self.item_samples = np.array(samples,dtype=int)
        self.item_sampled_vals = np.squeeze(np.array(sampled_vals))-1

        samples = []
        sampled_vals = []
        for node in range(self.adj_trans.shape[0]):
            neib = self.adj_trans[node]
            if neib.nnz >= self.layer_sizes:
                sample = np.random.choice(neib.indices, self.layer_sizes, replace=False)
                
            elif neib.nnz > 0:
                sample = np.random.choice(neib.indices, self.layer_sizes, replace=True)
            val = neib[0, sample].toarray()
            samples.append(sample)
            sampled_vals.append(val)
        self.user_samples = np.array(samples,dtype=int)
        self.user_sampled_vals = np.squeeze(np.array(sampled_vals))-1
        
