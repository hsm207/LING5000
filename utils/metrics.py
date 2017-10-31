import tensorflow as tf
from itertools import chain
from sklearn.metrics import f1_score
import pandas as pd
import os


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


def sk_f1_score(y, yhat, method):
    """
    Use scikit learn to compute the model's f1 micro/macro score
    :param y: A numeric tensor representing the ground truth
    :param yhat: A numeric tensor representing the mdodel's predictions
    :param method: A string to denote the type of f1 score to compute ('macro' or 'micro')
    :return: A double repsenting the metric computed
    """
    return f1_score(y, yhat, average=tf.compat.as_str(method))


def model_evaluation_metrics(categorical_labels, categorical_predictions):
    """
    Computes a dictionary of metrics to evaluate a model's preictions.

    :param categorical_labels: A tensor of group truths
    :param categorical_predictions: A tensor of predictions
    :return: A dictionary of metrics that can be passed to eval_metric_ops in the Estimator API
    """
    return {
        'accuracy': tf.metrics.accuracy(categorical_labels, categorical_predictions, name='metrics.accuracy'),
        'f1-macro': streaming_f1(categorical_labels, categorical_predictions, 11, 'macro'),
        'f1-micro': streaming_f1(categorical_labels, categorical_predictions, 11, 'micro')
    }


def consolidate_metrics(results_folder):
    """
    Consolidate all the csv files in the results folder
    :param results_folder: Path to the results folder
    :return: A data frame with each row containing the contents of the csv files in the results folder
    """
    return pd.concat([pd.read_csv(os.path.join(results_folder, file), index_col='model_name')
                      for file in os.listdir(results_folder)], axis=0)
