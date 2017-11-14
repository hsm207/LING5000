import tensorflow as tf
import numpy as np
from tensorflow.python.estimator.model_fn import EstimatorSpec
from utils.metrics import model_evaluation_metrics, sk_f1_score
from tensorflow.contrib.keras.api.keras.layers import Embedding, Conv2D, Flatten, concatenate, Dropout, Dense
from tensorflow.contrib.keras.api.keras.activations import softmax
from tensorflow.contrib.keras.api.keras.initializers import TruncatedNormal
from tensorflow.contrib.keras.api.keras.backend import set_learning_phase
from sklearn.metrics import accuracy_score


def cnn_1(features, labels, mode, params):
    """
    First naive CNN model.
    Pass this function to the model_fn of an Estimator

    """

    def convolution_builder(filters, kernel_sizes, strides, features):
        return [
            Conv2D(f, kernel_size=k, strides=s, padding='same', data_format='channels_last', activation='relu')(feat)
            for f, k, s, feat in zip(filters, kernel_sizes, strides, features)]

    n_gram = 2
    data_pad_size = 30
    pos_embedding_vocab = 36 + 1
    pos_embedding_dim = 48
    positional_embedding_vocab = 30 + 1
    positional_embedding_dim = 48

    if mode == tf.estimator.ModeKeys.TRAIN:
        set_learning_phase(True)
    else:
        set_learning_phase(False)

    # Grab the features
    tokens_features = features['token_idx']
    pos_features = features['pos_idx']
    positional_e1_features = features['positional_e1_idx']
    positional_e2_features = features['positional_e2_idx']

    with tf.name_scope('embedding_matrices'):
        # Build the embedding matrices
        word_embeddings_filename = params['word_embeddings_path']
        word_embeddings = np.genfromtxt(word_embeddings_filename, delimiter=' ',
                                        dtype=np.float32)

        word_embedding_vocab, word_embedding_dim = word_embeddings.shape

        word_embeddings = Embedding(word_embedding_vocab, word_embedding_dim, mask_zero=True,
                                    input_length=data_pad_size,
                                    weights=[word_embeddings],
                                    trainable=False,
                                    name='word_embdeddings_lookup_mat')

        pos_embeddings = Embedding(pos_embedding_vocab, pos_embedding_dim, mask_zero=True, input_length=data_pad_size,
                                   trainable=False,
                                   embeddings_initializer=TruncatedNormal,
                                   name='pos_embeddings')

        positional_e1_embeddings = Embedding(positional_embedding_vocab, positional_embedding_dim, mask_zero=True,
                                             input_length=data_pad_size,
                                             trainable=False,
                                             embeddings_initializer=TruncatedNormal,
                                             name='positional_e1_embeddings')

        positional_e2_embeddings = Embedding(positional_embedding_vocab, positional_embedding_dim, mask_zero=True,
                                             input_length=data_pad_size,
                                             trainable=False,
                                             embeddings_initializer=TruncatedNormal,
                                             name='positional_e2_embeddings')

    with tf.name_scope('embedding_lookups'):
        # Convert each feature into a vector representation
        tokens_vec = word_embeddings(tokens_features)
        # tokens_vec = tf.nn.embedding_lookup(word_embeddings, tokens_features)
        pos_vec = pos_embeddings(pos_features)
        positional_e1_vec = positional_e1_embeddings(positional_e1_features)
        positional_e2_vec = positional_e2_embeddings(positional_e2_features)

    with tf.name_scope('convolution_layers'):
        # The convolution layers
        conv1 = convolution_builder([4, 4, 4, 4],
                                    [(n_gram, word_embedding_dim),
                                     (n_gram, pos_embedding_dim),
                                     (n_gram, positional_embedding_dim),
                                     (n_gram, positional_embedding_dim)],
                                    [(1, word_embedding_dim), (1, pos_embedding_dim), (1, positional_embedding_dim),
                                     (1, positional_embedding_dim)],
                                    [tf.expand_dims(f, -1) for f in
                                     [tokens_vec, pos_vec, positional_e1_vec, positional_e2_vec]])

        conv2 = convolution_builder([8, 8, 8, 8],
                                    [(2, 1),
                                     (2, 1),
                                     (2, 1),
                                     (2, 1)],
                                    [(2, 1), (2, 1), (2, 1), (2, 1)],
                                    conv1)

        conv3 = convolution_builder([16, 16, 16, 16],
                                    [(3, 1),
                                     (3, 1),
                                     (3, 1),
                                     (3, 1)],
                                    [(3, 1), (3, 1), (3, 1), (3, 1)],
                                    conv2)

    with tf.name_scope('dense_layers'):
        tokens_conv, pos_conv, positional_e1_conv, positional_e2_conv = conv3

        tokens_flat = Flatten()(tokens_conv)
        pos_flat = Flatten()(pos_conv)
        positional_e1_flat = Flatten()(positional_e1_conv)
        positional_e2_flat = Flatten()(positional_e2_conv)

        merged_features = concatenate([tokens_flat, pos_flat, positional_e1_flat, positional_e2_flat])
        merged_features = Dropout(0.5)(merged_features)

        merged_features = Dense(merged_features.shape[1].value / 2, activation='relu')(merged_features)
        merged_features = Dropout(0.5)(merged_features)

        merged_features = Dense(merged_features.shape[1].value / 2, activation='relu')(merged_features)
        prediction_logits = Dense(11, activation='linear')(merged_features)
        prediction_softmax = softmax(prediction_logits)

    categorical_prediction = tf.argmax(prediction_softmax, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"relation": categorical_prediction})

    with tf.name_scope('training_step'):
        class_weights = params['class_weights']
        categorical_labels = tf.argmax(labels, 1)
        batch_weights = tf.gather(class_weights, categorical_labels)

        loss = tf.losses.sparse_softmax_cross_entropy(categorical_labels, prediction_logits, batch_weights)
        train_op = tf.train.AdamOptimizer().minimize(loss, tf.train.get_global_step())

    with tf.name_scope('metrics'):
        confusion_matrix = tf.confusion_matrix(categorical_labels, categorical_prediction)

        f1_macro = tf.py_func(sk_f1_score, [tf.argmax(labels, 1), tf.argmax(prediction_softmax, 1), 'macro'],
                              tf.float64, name='f1mac_metric')
        f1_micro = tf.py_func(sk_f1_score, [tf.argmax(labels, 1), tf.argmax(prediction_softmax, 1), 'micro'],
                              tf.float64, name='f1mic_metric')

        accuracy = tf.py_func(accuracy_score, [tf.argmax(labels, 1), tf.argmax(prediction_softmax, 1)], tf.float64,
                              name="accuracy")

    eval_metric_ops = model_evaluation_metrics(categorical_labels, categorical_prediction)

    return EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
