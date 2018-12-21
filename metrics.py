import tensorflow as tf

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def softmax_cross_entropy(preds, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)

def accuracy(preds, labels):
    #correct_prediction = tf.equal(preds, tf.cast(labels, tf.float32))
    
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)

def expected_rmse(logits, labels, class_values=None):
    """
    Computes the root mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth label
    :param class_values: rating values corresponding to each class.
    :return: rmse
    """

    probs = tf.nn.softmax(logits)
    if class_values is None:
        scores = tf.to_float(tf.range(start=0, limit=logits.get_shape()[1]) + 1)
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        scores = class_values
        y = tf.gather(class_values, labels)

    pred_y = tf.reduce_sum(probs * scores, 1)

    diff = tf.subtract(tf.cast(y, tf.float32), pred_y)
    exp_rmse = tf.square(diff)
    exp_rmse = tf.cast(exp_rmse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(exp_rmse))


def rmse(logits, labels, class_values=None):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    if class_values is None:
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        y = tf.gather(class_values, labels)

    pred_y = logits

    diff = tf.subtract(y, pred_y)
    mse = tf.square(diff)
    mse = tf.cast(mse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(mse))