from nltk import word_tokenize
import re
from string import punctuation
import numpy as np
import pandas as pd
from html import unescape

punctuation = punctuation.replace('<', '')
punctuation = punctuation.replace('>', '')


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

    words_between_entities_regex = r'{e1}[{punct}]? [{punct}]?(.*?){e2}'.format(**regex_params)
    words_between_entities = re.search(words_between_entities_regex, tagged_sent)
    # assert (words_between_entities is not None), 'Cannot find words betweeen {} and {}!'.format(e1, e2)
    words_between_entities = words_between_entities.group(2)

    # Check if there are other entities in between the e1 and e2
    words_between_entities = re.sub(general_entity_regex, r'\1', words_between_entities)

    words_between_entities = word_tokenize(words_between_entities)
    return pd.Series({'tokens': words_between_entities})
