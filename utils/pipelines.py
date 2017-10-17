import tensorflow as tf


def parse_input_file_line(filename):
    """
    Convenience function to parse a line in the files to be fed to the model.

    By construction, all files that are to be fed into the model consist of lines of space separated integers
    :param filename:  path to the file whose lines are to be processed
    :return: A tensor of integers
    """
    return tf.contrib.data.TextLineDataset(filename) \
        .map(lambda line: tf.string_split([line], ' ').values) \
        .map(lambda num: tf.string_to_number(num, tf.int32))


def generate_batched_dataset(input_files, pad_size, batch_size):
    """
    Use tensorflow's Dataset API to feed input into the model.
    :param input_files: A list of 4 strings containing the file paths to the target label, pos indices,
                        positional index for entity 1 and positional index for entity 2 (must be in this order!)
    :param pad_size: An integer to specify the number of 0s to pad the features
    :param batch_size: An integer to specify the mini batch size
    :return: A batached dataset
    """
    target, pos, rel_e1, rel_e2, = input_files

    target_labels = tf.contrib.data.Dataset.from_tensor_slices([target]) \
        .flat_map(parse_input_file_line)

    pos_tags = tf.contrib.data.Dataset.from_tensor_slices([pos]) \
        .flat_map(parse_input_file_line)

    positional_embeddings_e1 = tf.contrib.data.Dataset.from_tensor_slices([rel_e1]) \
        .flat_map(parse_input_file_line)

    positional_embeddings_e2 = tf.contrib.data.Dataset.from_tensor_slices([rel_e2]) \
        .flat_map(parse_input_file_line)

    batched_dataset = tf.contrib.data.Dataset.zip((pos_tags, positional_embeddings_e1,
                                                   positional_embeddings_e2, target_labels)) \
        .padded_batch(batch_size, padded_shapes=(pad_size, pad_size, pad_size, 11))

    return batched_dataset


def load_word_embeddings(file, vocab_size, embedding_dim):
    """
    Loads a pre-trained word embedding matrix into tensorflow

    :param file: Path to the file containing the word embedding. The content of the file is assumed to be
                space separated floats
    :param vocab_size: Number of rows in the embedding matrix.
                       This should be the number of unique words in your corpus + 1 (so that zero padded inputs
                       can be looked up too)
    :param embedding_dim: Number of columns in the embedding matrix
    :return: An vocab_size x embedding_dim matrix of floats
    """
    embed_matrix = tf.read_file(file)
    embed_matrix = tf.expand_dims(embed_matrix, 0)
    embed_matrix = tf.string_split(embed_matrix, ' \n').values
    embed_matrix = tf.string_to_number(embed_matrix, tf.float32)
    embed_matrix = tf.reshape(embed_matrix, [vocab_size, embedding_dim])

    return embed_matrix