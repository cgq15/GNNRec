from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inits import zeros, glorot

flags = tf.app.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., 
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False, 
                 sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                         dtype=tf.float32, 
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class MeanConvolve(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, 
                 support, #prob, self_features,
                 placeholders=None,
                 name = 'con',
                 sparse_support=True,
                 sparse_inputs=False, act=tf.nn.relu, bias=True,
                 dropout=0, featureless=False, **kwargs):
        super(MeanConvolve, self).__init__(**kwargs)

        self.dropout = dropout

        self.name = name
        self.act = act
        self.support = support
        self.sparse_support = sparse_support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['neigh_weights'] = glorot([input_dim, output_dim],
                                                        name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def call(self, self_vec, neigh_vec):

        neigh_vecs = tf.nn.dropout(neigh_vec, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vec, 1-self.dropout)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
       
        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
       
        # [nodes] x [out_dim]
        output = tf.add_n([from_self, from_neighs])

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class MaxConvolve(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, 
                 support, #prob, self_features,
                 placeholders=None,
                 name = 'con',
                 sparse_support=True,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 dropout=0, featureless=False, **kwargs):
        super(MaxConvolve, self).__init__(**kwargs)

        self.dropout = dropout

        self.name = name
        self.act = act
        self.support = support
        self.sparse_support = sparse_support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.hidden_dim = 64
        self.mlp_layers = []

        self.mlp_layers.append(Dense(input_dim=input_dim,
                                 output_dim=self.hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        with tf.variable_scope(self.name + '_vars'):
            self.vars['neigh_weights'] = glorot([self.hidden_dim, output_dim],
                                                        name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        self.input_dim = input_dim
        self.output_dim = output_dim
        if self.logging:
            self._log_vars()

    def call(self, self_vec, neigh_vec):

        neigh_vecs = tf.nn.dropout(neigh_vec, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vec, 1-self.dropout)
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        output = tf.add_n([from_self, from_neighs])

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)