import tensorflow as tf
import pandas as pd

from utils.pipelines import my_input_fn
from models.cnn import cnn_1
from models.res_cnn import res_cnn_1

from tensorflow.contrib.learn import Experiment, RunConfig
from tensorflow.contrib.keras.python.keras.backend import clear_session


def report_experiment(experiment):
    experiment.train()
    clear_session()
    eval_dict = experiment.evaluate()
    clear_session()

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


training_filenames = ['data/real/subtask_1.1/user_generated/train_target_labels.csv',
                      'data/real/subtask_1.1/user_generated/train_features_POS.csv',
                      'data/real/subtask_1.1/user_generated/train_relative_position_e1.csv',
                      'data/real/subtask_1.1/user_generated/train_relative_position_e2.csv',
                      'data/real/subtask_1.1/user_generated/train_tokens_int.csv']

test_filenames = ['data/real/subtask_1.1/user_generated/test_target_labels.csv',
                  'data/real/subtask_1.1/user_generated/test_features_POS.csv',
                  'data/real/subtask_1.1/user_generated/test_relative_position_e1.csv',
                  'data/real/subtask_1.1/user_generated/test_relative_position_e2.csv',
                  'data/real/subtask_1.1/user_generated/test_tokens_int.csv']

word_embeddings_path = 'data/real/subtask_1.1/user_generated/word_embeddings_glove_6b_50d.csv'

training_input_fn = my_input_fn(training_filenames, 1, 30, 982)
test_input_fn = my_input_fn(test_filenames, 0, 30, 246)

# feed_dict = {'class_weights': [4.15, 6.57, 5.43, 12.28, 7.77, 16.16, 12.93, 153.5, 122.80, 23.62, 61.40],
#              'word_embeddings_path': word_embeddings_path}

train_steps = 500
# Default hyper parameters for all models
feed_dict = {'class_weights': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             'word_embeddings_path': word_embeddings_path,
             'number_of_res_cnn_blocks': 4}

for i in range(1, 6):
    feed_dict['number_of_res_cnn_blocks'] = i
    config = RunConfig(model_dir='./model_dir/res_cnn_' + str(i))
    classifier_res_cnn = tf.estimator.Estimator(model_fn=res_cnn_1, params=feed_dict,
                                                config=config)
    experiment_res_cnn_1 = Experiment(
        estimator=classifier_res_cnn,
        train_input_fn=training_input_fn,
        eval_input_fn=test_input_fn,
        eval_metrics=None,
        train_steps=train_steps,
        min_eval_frequency=0,
        eval_delay_secs=0
    )

    res_cnn_df = report_experiment(experiment_res_cnn_1)
    res_cnn_df.to_excel('./results/' + res_cnn_df.index[0] + '.xlsx', 'results')

config_res_cnn_1 = RunConfig(model_dir='./model_dir/res_cnn_1')
classifier_res_cnn_1 = tf.estimator.Estimator(model_fn=res_cnn_1, params=feed_dict,
                                              config=config_res_cnn_1)

experiment_res_cnn_1 = Experiment(
    estimator=classifier_res_cnn_1,
    train_input_fn=training_input_fn,
    eval_input_fn=test_input_fn,
    eval_metrics=None,
    train_steps=train_steps,
    min_eval_frequency=0,
    eval_delay_secs=0
)

feed_dict.pop('number_of_res_cnn_blocks')
config_cnn_1 = RunConfig(model_dir='./model_dir/cnn_1')
classifier_cnn_1 = tf.estimator.Estimator(model_fn=cnn_1, params=feed_dict, model_dir='./model_dir/cnn_1',
                                          config=config_cnn_1)

experiment_cnn_1 = Experiment(
    estimator=classifier_cnn_1,
    train_input_fn=training_input_fn,
    eval_input_fn=test_input_fn,
    eval_metrics=None,
    train_steps=train_steps,
    min_eval_frequency=0,
    eval_delay_secs=0
)

cnn_1_df = report_experiment(experiment_cnn_1)

cnn_1_df.to_excel('./results/' + cnn_1_df.index[0] + '.xlsx', 'results')
