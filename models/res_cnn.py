from tensorflow.contrib.keras.api.keras.layers import Embedding, Convolution2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.contrib.keras.api.keras.initializers import Constant
from tensorflow.contrib.keras.api.keras.activations import softmax
from tensorflow.contrib.keras.api.keras.backend import set_learning_phase
from tensorflow.python.estimator.model_fn import EstimatorSpec
from utils.metrics import model_evaluation_metrics
import tensorflow as tf
import numpy as np


def res_cnn_1(features, labels, mode, params):
    """
    First implemention of a convolutional neural network with residual connections.

    Based on https://arxiv.org/pdf/1707.08866.pdf

    """

    def residual_cnn_block_builder(block_input, block_id, filter_height):
        block_name = 'B{}_residual_cnn_block'.format(block_id)
        conv1_name = 'B{}_convolution1'.format(block_id)
        conv2_name = 'B{}_convolution2'.format(block_id)
        identity_shortcut_name = 'B{}_identity_chortcut'.format(block_id)

        with tf.name_scope(block_name):
            conv1 = Convolution2D(filters=n_filters, kernel_size=[filter_height, block_input.shape[2].value], strides=1,
                                  padding='same',
                                  data_format='channels_last', activation='relu', name=conv1_name)(block_input)

            conv2 = Convolution2D(filters=n_filters, kernel_size=[filter_height, conv1.shape[2].value], strides=1,
                                  padding='same',
                                  data_format='channels_last', activation='relu', name=conv2_name)(conv1)

            identity_shortcut = tf.add(conv2, block_input, name=identity_shortcut_name)

        return identity_shortcut

    class_weights_params = params['class_weights']
    word_embeddings_file_path_param = params['word_embeddings_path']
    n_res_cnn_blocks_param = params['number_of_res_cnn_blocks']
    n_filters = params['number_of_filters']
    other_feature_dim = params['other_feature_embeedding_dim']
    filter_height = params['filter_height']

    if mode == tf.estimator.ModeKeys.TRAIN:
        set_learning_phase(True)
    else:
        set_learning_phase(False)

    with tf.name_scope('input_layer'):
        pos_embedding_vocab = 40 + 1
        pos_embedding_dim = other_feature_dim
        positional_embedding_vocab_e1 = 165 + 1
        positional_embedding_vocab_e2 = 166 + 1
        positional_embedding_dim = other_feature_dim

        pos_idx = features['pos_idx']
        positional_e1_idx = features['positional_e1_idx']
        positional_e2_idx = features['positional_e2_idx']
        token_idx = features['token_idx']

        word_embeddings_filename = word_embeddings_file_path_param
        word_embeddings = np.genfromtxt(word_embeddings_filename, delimiter=' ',
                                        dtype=np.float32)
        word_embedding_vocab, word_embedding_dim = word_embeddings.shape

        word_embeddings = Embedding(word_embedding_vocab, word_embedding_dim, input_length=token_idx.shape[1].value,
                                    embeddings_initializer=Constant(word_embeddings),
                                    name='word_embeddings')
        pos_embeddings = Embedding(pos_embedding_vocab, pos_embedding_dim, input_length=pos_idx.shape[1].value,
                                   name='part-of-speech_embeddings')
        positional_e1_embeddings = Embedding(positional_embedding_vocab_e1, positional_embedding_dim,
                                             input_length=positional_e1_idx.shape[1].value,
                                             name='entity1_positional_embedding')
        positional_e2_embeddings = Embedding(positional_embedding_vocab_e2, positional_embedding_dim,
                                             input_length=positional_e2_idx.shape[1].value,
                                             name='entity2_positional_embedding')

        word_vec = word_embeddings(token_idx)
        pos_vec = pos_embeddings(pos_idx)
        positional_e1_vec = positional_e1_embeddings(positional_e1_idx)
        positional_e2_vec = positional_e2_embeddings(positional_e2_idx)

        features_concat = tf.concat([word_vec, pos_vec, positional_e1_vec, positional_e2_vec], 2,
                                    name='concatenated_features')

        # make features_concat to have rank 4 (batch_size, row_size, column_size, channel_size)
        features_concat = tf.expand_dims(features_concat, -1)

    with tf.name_scope('convolution_layer'):
        features_conv = Convolution2D(filters=n_filters, kernel_size=[filter_height, features_concat.shape[2].value],
                                      strides=1,
                                      padding='valid',
                                      data_format='channels_last', activation='relu')(features_concat)

    with tf.name_scope('residual_CNN_blocks_layer'):
        for i in range(n_res_cnn_blocks_param):
            block_output = residual_cnn_block_builder(features_conv, i, filter_height)
            features_conv = block_output + features_conv

    with tf.name_scope('pooling_layer'):
        features_pooled = MaxPool2D(pool_size=features_conv.shape[1:3].as_list(), padding='valid',
                                    data_format='channels_last')(features_conv)

    with tf.name_scope('dense_layers'):
        features_flat = Flatten()(features_pooled)
        features_flat = Dropout(0.5)(features_flat)
        dense1 = Dense(n_filters, activation='relu', name='dense_1')(features_flat)
        dense2 = Dense(n_filters, activation='relu', name='dense_2')(dense1)

    with tf.name_scope('output_layer'):
        output_logits = Dense(11, activation='linear', name='logits')(dense2)
        output_softmax = softmax(output_logits)
        output_predictions = tf.argmax(output_softmax, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'relation': output_predictions,
                         'id': tf.squeeze(features['id'])})

    with tf.name_scope('training_step'):
        class_weights = class_weights_params
        ground_truths = tf.argmax(labels, axis=1)
        batch_weights = tf.gather(class_weights, ground_truths)

        loss = tf.losses.sparse_softmax_cross_entropy(ground_truths, output_logits, batch_weights)
        train_op = tf.train.AdamOptimizer().minimize(loss, tf.train.get_global_step())

    with tf.name_scope('metrics'):
        eval_metric_ops = model_evaluation_metrics(ground_truths, output_predictions)

    return EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
