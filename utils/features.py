from utils.parse_data import extract_entities
from nltk import word_tokenize, StanfordPOSTagger
import os
import re
from string import punctuation
import numpy as np
import pandas as pd
from html import unescape

punctuation = punctuation.replace('<', '')
punctuation = punctuation.replace('>', '')


def get_entity_word_offsets(tagged_sentence, target_entity_id):
    """
    Generate the word offsets in a tagged sentence for a given entity id
    :param tagged_sentence: A string containing a sentence with tagged entities
    :param target_entity_id: A string representing the entity's id that we can to compute the offsets from
    :return: A numpy array of integers representing the offset of each word in the sentence relative to the target
             entity
    """

    def entity_regex_generator(entity_id):
        regex = r'<entity id="{}">(?P<entity_text>.*?)</entity>'.format(entity_id)
        return regex

    def convert_tagged_entity_to_string(sentence, entity_tag):
        return re.sub(entity_tag, r'\g<entity_text>', sentence)

    tagged_target_entity = entity_regex_generator(target_entity_id)
    regex_words_around_entity = f"(?P<lhs_words>.*)?{tagged_target_entity}(?P<rhs_words>.*)?"
    other_entities = (k for k in extract_entities(tagged_sentence).keys() if k != target_entity_id)

    for entity in other_entities:
        tagged_sentence = convert_tagged_entity_to_string(tagged_sentence, entity_regex_generator(entity))

    lhs_words, entity_words, rhs_words = (word_tokenize(sent) for sent in
                                          re.search(regex_words_around_entity, tagged_sentence).groups())

    i_lhs_words, i_rhs_words = ([i for i, _ in enumerate(words, 1)] for words in (lhs_words, rhs_words))

    if i_lhs_words != []:
        i_lhs_words.reverse()
        i_lhs_words = -1 * np.array(i_lhs_words)

    position_embedding = np.concatenate([i_lhs_words,
                                         np.zeros(len(entity_words)),
                                         i_rhs_words]).astype(np.int32)
    return position_embedding

def get_words_between_entities(row):
    """
    A function to extract the words in between an entity pair. This function is meant to be used via pandas apply()
    function on a dataframe that contains ONLY the following columns:
        1. Tagged sentence: A sentence in the corpus with all of the xml markups
        2. Entity1: The entity ID for the first entity
        3. Entity2: The entity ID for the second entity

    :param row: A row from the pandas dataframe that is passed to the call to apply()
    :return: A single column pandas dataframe where each row contains a list of words in between the
             corresponding entity pairs
    """

    def entity_regex_generator(entity_id):
        regex = r'<entity id="{}">(.*?)</entity>'.format(entity_id)
        return regex

    tagged_sent, e1, e2 = row
    tagged_sent = unescape(tagged_sent)  # some sentences e.g. text_id C04-1036 have html entities in them e.g. C04-1036
    e1_regex = entity_regex_generator(e1)
    e2_regex = entity_regex_generator(e2)
    general_entity_regex = entity_regex_generator('.*?')

    regex_params = {
        'e1': e1_regex,
        'e2': e2_regex,
        'punct': punctuation
    }

    words_between_entities_regex = r'{e1}(.*?){e2}'.format(**regex_params)
    words_between_entities = re.search(words_between_entities_regex, tagged_sent)
    # assert (words_between_entities is not None), 'Cannot find words betweeen {} and {}!'.format(e1, e2)
    words_between_entities = words_between_entities.group(2)

    # Check if there are other entities in between the e1 and e2
    words_between_entities = re.sub(general_entity_regex, r'\1', words_between_entities)

    words_between_entities = word_tokenize(words_between_entities)
    return pd.Series({'tokens': words_between_entities})


def get_positional_embeddings(row):
    """
    Compute the positional embeddings given a list of words that are between a pair of entities. This function is
    meant to be called via a panda's dataframe.apply() function where the dataframe is a single column named
     'tokens' containing the list of words.

    :param row: A row from the pandas dataframe that is passed to the call to apply()
    :return: A two column dataframe where column 1 is the positional embedding relative to entity 1 and column 2
             is the positional embedding relative to entity 2
    """
    word_list = row['tokens']
    n_words_between_entities = len(word_list)

    relative_position_e1 = np.arange(1, n_words_between_entities + 1)
    relative_position_e2 = np.flipud(-np.arange(1, n_words_between_entities + 1))

    return pd.Series({'relative_position_e1': relative_position_e1,
                      'relative_position_e2': relative_position_e2})


def initialize_pos_tagger(nthreads='1',
                          model='wsj-0-18-bidirectional-distsim.tagger',
                          java_home='C:/ProgramData/Oracle/Java/javapath/java.exe',
                          classpath='resources/stanford-postagger-full-2017-06-09',
                          tagger_models='resources/stanford-postagger-full-2017-06-09/models'):
    """
    This function initializes the Stanford POS tagger with the given parameters and returns a function that will
    perform the POS-tagging with parameterized tagger.

    The input to the returned function is a list of a list of tokens and the output is a list of corresponding POS tag.
    :param nthreads: A string representing the number of threads the tagger should be initialized with
    :param model: A string representing one of Stanford's pre-trained taggers
    :param java_home: A string representing the full path to the java executable
    :param classpath: A string representing the path to the stanford-postagger.jar file
    :param tagger_models: A string representing the path to the models folder containing the pre-trained models
    :return: A function that takes as input a list of a list of tokens and returns a list of the corresponding POS tags
    """
    os.environ['JAVA_HOME'] = java_home
    os.environ['CLASSPATH'] = classpath
    os.environ['STANFORD_MODELS'] = tagger_models

    # Subclass the StanfordPOSTagger to add the nthreads option and customize its output
    class CustomStanfordPOSTagger(StanfordPOSTagger):
        def __init__(self, *args, **kwargs):
            if 'nthreads' in kwargs:
                self.nthreads = kwargs.pop('nthreads')

            else:
                self.nthreads = '1'

            super().__init__(*args, **kwargs)

        @property
        def _cmd(self):
            custom_cmd = ['-nthreads', self.nthreads]
            return super()._cmd + custom_cmd

        def parse_output(self, text, sentences=None):
            # Output the tagged sentences
            tagged_sentences = []
            for tagged_sentence in text.strip().split("\n"):
                sentence = []
                for tagged_word in tagged_sentence.strip().split():
                    word_tags = tagged_word.strip().split(self._SEPARATOR)
                    sentence.append(word_tags[-1])
                tagged_sentences.append(sentence)
            return tagged_sentences

    pt = CustomStanfordPOSTagger(model, verbose=True, java_options='-mx2G', nthreads=nthreads)

    def get_pos_tags(tokens_list):
        return pt.tag_sents(tokens_list)

    return get_pos_tags
