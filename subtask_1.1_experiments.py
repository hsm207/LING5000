import tensorflow as tf
import pandas as pd
import numpy as np
from itertools import product

from utils.pipelines import my_input_fn
from models.cnn import cnn_1
from models.res_cnn import res_cnn_1

from tensorflow.contrib.learn import Experiment, RunConfig
from tensorflow.contrib.keras.python.keras.backend import clear_session


def report_experiment(experiment):
    # experiment.train()
    # clear_session()
    eval_dict, _ = experiment.train_and_evaluate()
    # clear_session()

    model_name = experiment.estimator.model_dir.split('/')[2]
    model_params = experiment.estimator.params

    model_params['word_embeddings_path'] = model_params['word_embeddings_path'].split('/')[4]
    model_params['class_weights'] = str(model_params['class_weights'])

    eval_df = pd.DataFrame.from_dict(eval_dict, orient='index').transpose()
    params_df = pd.DataFrame.from_dict(model_params, orient='index').transpose()

    report_df = pd.concat([eval_df, params_df], axis=1)
    report_df['model_name'] = model_name
    report_df.set_index('model_name', inplace=True)

    return report_df


training_filenames = ['data/real/subtask_1.1/user_generated/train_ids.csv',
                      'data/real/subtask_1.1/user_generated/train_target_labels.csv',
                      'data/real/subtask_1.1/user_generated/train_features_POS.csv',
                      'data/real/subtask_1.1/user_generated/train_relative_position_e1.csv',
                      'data/real/subtask_1.1/user_generated/train_relative_position_e2.csv',
                      'data/real/subtask_1.1/user_generated/train_tokens_int.csv']

test_filenames = ['data/real/subtask_1.1/user_generated/test_ids.csv',
                  'data/real/subtask_1.1/user_generated/test_target_labels.csv',
                  'data/real/subtask_1.1/user_generated/test_features_POS.csv',
                  'data/real/subtask_1.1/user_generated/test_relative_position_e1.csv',
                  'data/real/subtask_1.1/user_generated/test_relative_position_e2.csv',
                  'data/real/subtask_1.1/user_generated/test_tokens_int.csv']

predict_filenames = ['data/real/subtask_1.1/user_generated/consol_ids.csv',
                     'data/real/subtask_1.1/user_generated/consol_target_labels.csv',
                     'data/real/subtask_1.1/user_generated/consol_features_POS.csv',
                     'data/real/subtask_1.1/user_generated/consol_relative_position_e1.csv',
                     'data/real/subtask_1.1/user_generated/consol_relative_position_e2.csv',
                     'data/real/subtask_1.1/user_generated/consol_tokens_int.csv']

word_embeddings_path = 'data/real/subtask_1.1/user_generated/word_embeddings_glove_6b_50d.csv'

training_input_fn = my_input_fn(training_filenames, 1, 101, 982)
test_input_fn = my_input_fn(test_filenames, 0, 101, 246)
predict_input_fn = my_input_fn(predict_filenames, 0, 30, 982 + 246)

train_steps = [200, 600, 800, 1000, 1200, 1500]
# Default hyper parameters for all models
feed_dict = {'class_weights': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             'word_embeddings_path': word_embeddings_path,
             'number_of_res_cnn_blocks': 2,
             'filter_size': 128}

# hyperparameters for class weights
class_weights_ratios = np.arange(1, 5.5, 1.5)
class_freq = np.array([0.08143322, 0.24104235, 0.00814332, 0.06188925, 0.04234528,
                       0.1286645, 0.07736156, 0.01628664, 0.18403909, 0.15228013,
                       0.00651466])
max_freq = max(class_freq)
max_arg = np.argmax(class_freq)

class_weights_params = []
class_weights_params.append(1 / class_freq)
class_weights_params.append(np.repeat(1, 11))

for r in class_weights_ratios:
    weights = (r * max_freq) / class_freq
    weights[max_arg] = 1
    class_weights_params.append(weights)
    class_weights_params.append(weights / np.sum(weights))

class_weights_params = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]  # use equal weights until can figure method to pick weights?

# hyper parameters for number of cnn blocks
n_res_cnn_blocks = [0, 1, 2, 3, 4]

# hyper parameters for number of filters
n_filter_sizes = [128, 256, 512, 1024]
# combine all possible hyperparams
hyperparams = product(class_weights_params, n_res_cnn_blocks, n_filter_sizes)
hyperparams = [([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0, 128)]
train_steps = [5, 10, 15]
for i, (class_weights, n_res_cnn, n_filter_size) in enumerate(hyperparams, 1):

    for epochs in train_steps:
        feed_dict['class_weights'] = class_weights
        feed_dict['number_of_res_cnn_blocks'] = n_res_cnn
        feed_dict['filter_size'] = n_filter_size

        config = RunConfig(model_dir='./model_dir/res_cnn_1_' + str(i), save_summary_steps=1)
        classifier_res_cnn = tf.estimator.Estimator(model_fn=res_cnn_1, params=feed_dict,
                                                    config=config)
        experiment_res_cnn_1 = Experiment(
            estimator=classifier_res_cnn,
            train_input_fn=training_input_fn,
            eval_input_fn=test_input_fn,
            eval_metrics=None,
            train_steps=epochs,
            min_eval_frequency=0,  # only evaluate after done with training
            eval_delay_secs=0
        )

        # eval_result, _ = experiment_res_cnn_1.train_and_evaluate()
        res_cnn_df = report_experiment(experiment_res_cnn_1)
        res_cnn_df.to_csv('./results/' + res_cnn_df.index[0] + '_{}.csv'.format(epochs))

print('Finished experiments!')
