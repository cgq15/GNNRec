import layers
import metrics
import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.reg_loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()


        # Build sequential layer model
        self.activations.append(self.inputs)

        # uncomment the following to enable skip connection
        # hidden_12, reg_loss_12 = self.layers[0](self.inputs)
        # hidden_23, reg_loss_23 = self.layers[1](hidden_12)
        # self.outputs = hidden_23
        # hidden_13, reg_loss_13 = self.layers[2](self.inputs)
        # self.outputs += FLAGS.skip*hidden_13
        # self.reg_loss = reg_loss_23
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    """ A standard multi-layer perceptron """
    def __init__(self, placeholders, dims, cross_en=True, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.placeholders = placeholders
        self.cross_en = cross_en
        self.inputs = placeholders['features']
        self.labels = tf.cast(placeholders['labels'],tf.float32)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        if self.cross_en:
            self.loss += metrics.rmse(self.outputs, self.labels)
        # L2
        else:
            diff = self.labels - self.outputs
            self.loss += tf.reduce_sum(tf.sqrt(tf.reduce_sum(diff * diff, axis=1)))

    def _accuracy(self):
        pass
        #if self.cross_en:
        #    self.accuracy = metrics.accuracy(self.outputs, self.labels)

    def _build(self):
        self.layers.append(layers.Dense(input_dim=self.input_dim,
                                 output_dim=self.dims[1],
                                 act=tf.nn.relu,
                                 dropout=self.placeholders['dropout'],
                                 sparse_inputs=False,
                                 logging=self.logging))
        self.layers.append(layers.Dense(input_dim=self.dims[1],
                                 output_dim=self.dims[2],
                                 act=tf.nn.relu,
                                 dropout=self.placeholders['dropout'],
                                 sparse_inputs=False,
                                 logging=self.logging))                        

        self.layers.append(layers.Dense(input_dim=self.dims[2],
                                 output_dim=self.output_dim,
                                 act=lambda x: x,
                                 dropout=self.placeholders['dropout'],
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method

    (Removes the layers idiom from build method of the Model class)
    """

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)
        

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

class GNN(GeneralizedModel):
    def __init__(self, placeholders, u_features, v_features, class_values, dims,
            concat=True, identity_dim=0, cross_en=True, **kwargs):
        super(GNN, self).__init__(**kwargs)
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        self.v_supports = placeholders["item_support"]
        self.u_supports = placeholders["user_support"]
        self.labels = placeholders['labels']
        self.dropout = placeholders['dropout']
        self.class_values = class_values
        self.cross_en = cross_en

        if identity_dim > 0:
            self.user_embeds = tf.get_variable("user_embeddings", [FLAGS.usernum, identity_dim])
            self.item_embeds = tf.get_variable("item_embeddings", [FLAGS.itemnum, identity_dim])
        else:
            self.user_embeds = None
            self.item_embeds = None
        
        if FLAGS.features is False: 
            if FLAGS.identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.u_features = self.user_embeds
            self.v_features = self.item_embeds
        else:
            self.u_features = tf.Variable(tf.constant(u_features, dtype=tf.float32), trainable=False)
            self.v_features = tf.Variable(tf.constant(v_features, dtype=tf.float32), trainable=False)
            if not self.user_embeds is None:
                self.u_features = tf.concat([self.user_embeds, self.u_features], axis=1)
                self.v_features = tf.concat([self.item_embeds, self.v_features], axis=1)
        self.u_dim = self.u_features.shape[1]
        self.v_dim = self.v_features.shape[1]
        
        self.item_input = tf.nn.embedding_lookup(self.v_features, self.v_indices)
        self.user_input = tf.nn.embedding_lookup(self.u_features, self.u_indices)
        self.u_support_input = tf.nn.embedding_lookup(self.u_features, self.u_supports)
        self.v_support_input = tf.nn.embedding_lookup(self.v_features, self.v_supports)
        self.Wv = tf.get_variable("Wv", shape=(self.v_dim, dims[1]), dtype=tf.float32, 
                                         initializer=tf.contrib.layers.xavier_initializer())
        self.Wu = tf.get_variable("Wu", shape=(self.u_dim, dims[1]), dtype=tf.float32, 
                                         initializer=tf.contrib.layers.xavier_initializer())
        self.Wout = tf.get_variable("Wout", shape=(dims[1]+dims[1], dims[-1]), dtype=tf.float32, 
                                         initializer=tf.contrib.layers.xavier_initializer())                                 
        self.v_aggregator = layers.MeanConvolve(input_dim=dims[1],
                                 output_dim=dims[1],
                                 act=tf.nn.relu,
                                 dropout=placeholders['dropout'],
                                 name='vcon',
                                 support=self.v_supports,
                                 sparse_inputs=False,
                                 logging=self.logging)
        self.u_aggregator = layers.MeanConvolve(input_dim=dims[1],
                                 output_dim=dims[1],
                                 act=tf.nn.relu,
                                 dropout=placeholders['dropout'],
                                 name='ucon',
                                 support=self.u_supports,
                                 sparse_inputs=False,
                                 logging=self.logging)                         
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.user_vecs = None
        self.item_vecs = None
        self.item_support_val = placeholders['item_support_val']
        self.user_support_val = placeholders['user_support_val']
        self.i_edge_weights = tf.get_variable("i_edge_weight", shape=(FLAGS.classnum, dims[1]), dtype=tf.float32, 
                                         initializer=tf.contrib.layers.xavier_initializer())
        self.u_edge_weights = tf.get_variable("u_edge_weight", shape=(FLAGS.classnum, dims[1]), dtype=tf.float32, 
                                         initializer=tf.contrib.layers.xavier_initializer())                                 

        self.build()
    

    def _build(self):
        user_vec = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.user_input, self.Wu)),1)
        item_vec = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.item_input, self.Wv)),1)

        v_support_input = tf.nn.relu(tf.tensordot(self.v_support_input, self.Wv, axes=1))
        u_support_input = tf.nn.relu(tf.tensordot(self.u_support_input, self.Wu, axes=1))

        v_support_vecs = tf.nn.l2_normalize(v_support_input,1)
        v_support_weights = tf.gather(self.i_edge_weights, self.item_support_val)
        v_support_vecs = v_support_vecs * v_support_weights

        u_support_vecs = tf.nn.l2_normalize(u_support_input,1)
        u_support_weights = tf.gather(self.u_edge_weights, self.user_support_val)
        u_support_vecs = u_support_vecs * u_support_weights

        user_vec = tf.nn.l2_normalize(self.v_aggregator.call(user_vec, v_support_vecs),1)  
        item_vec = tf.nn.l2_normalize(self.u_aggregator.call(item_vec, u_support_vecs),1)

        self.outputs = tf.matmul(tf.concat([user_vec, item_vec], -1), self.Wout)
        
    def _loss(self):
        # Weight decay loss
        '''
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        '''
        # Cross entropy error
        
        if self.cross_en:
            self.loss += metrics.expected_rmse(self.outputs, self.labels, self.class_values)
        # L2
        else:
            diff = self.labels - self.outputs
            self.loss += tf.reduce_sum(tf.sqrt(tf.reduce_sum(diff * diff, axis=1)))
        
    def _accuracy(self):
        return 0