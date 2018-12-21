from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import time
import os
from inits import *
from sampler import *
from utils import *
from model import MLP, GNN
from minibatch import MinibatchIter
import os
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'ml_100k', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_adapt', 'Model string.')  # 'gcn', 'gcn_appr'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('sample_num', 16, 'Maximum Chebyshev for constructing the adjacent matrix.')
flags.DEFINE_integer('gpu', '4', 'The gpu to be applied.')
flags.DEFINE_integer('batchsize', 256, 'The number of batchsize.')
flags.DEFINE_integer('classnum', 5, 'The number of classes.')
flags.DEFINE_bool('features', True, 'If use features')
flags.DEFINE_integer('usernum', 0, 'The number of users.')
flags.DEFINE_integer('itemnum', 0, 'The number of items.')
flags.DEFINE_integer('layer_num', 1, 'The number of items.')
flags.DEFINE_integer('identity_dim', 16, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

# Load data
os.environ["CUDA_VISIBLE_DEVICES"]='0,2'

def main():

    # Prepare data
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
    val_labels, val_u_indices, val_v_indices, test_labels, \
    test_u_indices, test_v_indices, class_values =  prepare_ml(FLAGS.dataset)
    FLAGS.usernum = u_features.shape[0]
    FLAGS.itemnum = v_features.shape[0]
    print('preparation done!')

    max_degree = FLAGS.sample_num
    num_train = adj_train.shape[0]-1
    scope = 'test'
    input_dim = 0
    num_users = u_features.shape[0]
    num_items = v_features.shape[0]
    if not FLAGS.features:
        u_features = sp.identity(num_users, format='csr')
        v_features = sp.identity(num_items, format='csr')
    
    elif FLAGS.features and u_features is not None and v_features is not None:
        input_dim = u_features.shape[1]+v_features.shape[1]

    else:
        raise ValueError('Features flag is set to true but no features are loaded from dataset ' + FLAGS.dataset)
    print(u_features.shape, v_features.shape)
    
    propagator = GNN
    # Define placeholders
    placeholders = {
        'batch': tf.placeholder(tf.int32),
        'labels': tf.placeholder(tf.int32, shape=(None,)),
        'item_support': tf.placeholder(tf.int32, shape=(None,max_degree,)),
        'item_support_val': tf.placeholder(tf.int32, shape=(None,max_degree,)),
        'user_support': tf.placeholder(tf.int32, shape=(None,max_degree,)),
        'user_support_val': tf.placeholder(tf.int32, shape=(None,max_degree,)),
        'user_indices': tf.placeholder(tf.int32, shape=(None,)),
        'item_indices': tf.placeholder(tf.int32, shape=(None,)),

        'dropout': tf.placeholder_with_default(0., shape=()),
    }



    val_feed_dict = {placeholders['labels']:val_labels, placeholders['user_indices']:val_u_indices,
                    placeholders['item_indices']:val_v_indices}

    test_feed_dict = {placeholders['labels']:test_labels, placeholders['user_indices']:test_u_indices,
                    placeholders['item_indices']:test_v_indices}
    # Sampling parameters shared by the sampler and model
    # Create sampler
    
    sampler_tf = SimpleSampler(placeholders, adj=adj_train, input_dim=input_dim, layer_sizes=max_degree)
    sampler_tf.gen_sample_list()
    # Create model
    model = propagator(placeholders, u_features.toarray(), v_features.toarray(), class_values,
            dims=[input_dim, FLAGS.hidden1, FLAGS.classnum], identity_dim = FLAGS.identity_dim, logging=True, name=scope)

    # Initialize session
    config = tf.ConfigProto(log_device_placement=False)
    sess = tf.Session(config=config)

    # Init variables
    sess.run(tf.global_variables_initializer())

    # Prepare training
    saver = tf.train.Saver()
    save_dir = "tmp/" + FLAGS.dataset + '_' + str(FLAGS.gpu)
    acc_val = []
    
    train_time = []
    train_time_sample = []
    best_val = 2

    minibatch = MinibatchIter(train_labels, train_u_indices, train_v_indices, FLAGS.batchsize)
    val_samples, val_sampled_val = sampler_tf.sampling(val_u_indices, 'item')
    val_feed_dict.update({placeholders['item_support']:val_samples, placeholders['item_support_val']:val_sampled_val})
    val_samples, val_sampled_val = sampler_tf.sampling(val_u_indices, 'user')
    val_feed_dict.update({placeholders['user_support']:val_samples, placeholders['user_support_val']:val_sampled_val})

    test_samples, test_sampled_val = sampler_tf.sampling(test_u_indices, 'item')
    test_feed_dict.update({placeholders['item_support']:test_samples, placeholders['item_support_val']:test_sampled_val})
    test_samples, test_sampled_val = sampler_tf.sampling(test_u_indices, 'user')
    test_feed_dict.update({placeholders['user_support']:test_samples, placeholders['user_support_val']:test_sampled_val})
    # Train model
    for epoch in range(FLAGS.epochs):
        loss_train = []
        sample_time = 0
        t1 = time.time()
        t = 0
        sampler_tf.gen_sample_list()
        minibatch.shuffle()
        while not minibatch.end():
            batch_size, batch_u, batch_v, batch_label = minibatch.next_minibatch()
            # print("batch: ", batch_size)
            t_sample = time.time()
            item_samples, item_sampled_val = sampler_tf.sampling(batch_u, 'item')
            user_samples, user_sampled_val = sampler_tf.sampling(batch_u, 'user')
            #print(batch_label, batch_features[0])
            #print (samples.shape, sampled_val.shape)
            feed_dict={placeholders['batch']:batch_size, placeholders['user_indices']:batch_u,
                         placeholders['item_indices']:batch_v, placeholders['labels']:batch_label,
                         placeholders['item_support']:item_samples, placeholders['item_support_val']:item_sampled_val,
                         placeholders['user_support']:user_samples, placeholders['user_support_val']:user_sampled_val}

            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
            #print(time.time()-t_sample)
            loss_train.append(outs[1])

        
        # Validation
        #cost, acc, duration = increment_evaluate(features, adj, test_index, y_test, [], placeholders)
        #acc_val.append(acc)
        
        val_loss= sess.run([model.loss], feed_dict=val_feed_dict)[0]
        if val_loss <= best_val:
            best_val = val_loss
            test_loss= sess.run([model.loss], feed_dict=test_feed_dict)
        # if epoch > 50 and acc>max_acc:
        #     max_acc = acc
        #     saver.save(sess, save_dir + ".ckpt")
        train_time_sample.append(time.time()-t1)
        train_time.append(time.time()-t1-sample_time)
        loss_train = np.mean(loss_train)
        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_train),
               "val_loss=", "{:.5f}".format(val_loss),
               "time=", "{:.5f}".format(train_time_sample[epoch]))

    train_duration = np.mean(np.array(train_time_sample))
    # Testing
    print("test loss=", "{:.5f}".format(test_loss[0]), "training time per epoch=", "{:.5f}".format(train_duration))



if __name__ == "__main__":

    print("DATASET:", FLAGS.dataset)
    main()