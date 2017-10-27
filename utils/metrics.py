import tensorflow as tf
from itertools import chain


def streaming_f1(labels, preds, n_classes, type='macro'):
    """
    A streaming metric to compute a model's macro/micro f1-score.

    :param labels: A numeric tensor representing the ground truth
    :param preds: A numeric tensor representing the mdodel's predictions
    :param n_classes: An integer representing the number of classes to be predicted
    :param type: A string to denote the type of f1 score to compute ('macro' or 'micro')
    :return: A tuple of tensors representing the metric's value and update op
    """
    labels_and_predictions_by_class = [(tf.equal(labels, c), tf.equal(preds, c)) for c in range(0, n_classes)]
    tp_by_class_val, tp_by_class_update_op = zip(
        *[tf.metrics.true_positives(labels, preds) for labels, preds in labels_and_predictions_by_class])
    fn_by_class_val, fn_by_class_update_op = zip(
        *[tf.metrics.false_negatives(labels, preds) for labels, preds in labels_and_predictions_by_class])
    fp_by_class_val, fp_by_class_update_op = zip(
        *[tf.metrics.false_positives(labels, preds) for labels, preds in labels_and_predictions_by_class])

    f1_update_op = tf.group(*chain(tp_by_class_update_op, fn_by_class_update_op, fp_by_class_update_op))

    if type == 'macro':
        epsilon = [10e-6 for _ in range(n_classes)]

        f1_val = tf.multiply(2., tp_by_class_val) / (tf.reduce_sum([tf.multiply(2., tp_by_class_val),
                                                                    fp_by_class_val, fn_by_class_val, epsilon],
                                                                   axis=0))
        f1_val = tf.reduce_mean(f1_val)
    else:
        epsilon = 10e-6

        total_tp = tf.reduce_sum(tp_by_class_val)
        total_fn = tf.reduce_sum(fn_by_class_val)
        total_fp = tf.reduce_sum(fp_by_class_val)

        f1_val = tf.squeeze(tf.multiply(2., total_tp) / (tf.multiply(2., total_tp) +
                                                         total_fp + total_fn + epsilon,
                                                         ))

    return f1_val, f1_update_op
