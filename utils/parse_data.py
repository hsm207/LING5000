import xml.etree.ElementTree as ET
from collections import OrderedDict
import nltk.data
import re
import pandas as pd
import numpy as np

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sent_detector._params.abbrev_types.add('viz')
sent_detector._params.abbrev_types.add('e.g')
sent_detector._params.abbrev_types.add('i.e')
sent_detector._params.abbrev_types.add('al')

entity_regex = re.compile(r'<entity id=\"([A-Z][\d.-]+)\">(.*?)</entity>')
abstract_regex = re.compile(r'<abstract>((?:.|\s)*)</abstract>')
relation_regex = re.compile(r'(?P<relation>[A-Z]+)\((?P<entity1>.+?),(?P<entity2>.+?)(?:,(?P<is_reversed>.+?))?\)')


def extract_entities(tagged_sentence):
    """
    Use regular expression to generate a dictionary of entities from a tagged sentence.
    :param tagged_sentence: A string representing a sentence with entity id tags in it
    :return: An ordered dictionary of the form {entity id: entity name}
    """
    entities = re.findall(entity_regex, tagged_sentence)
    return OrderedDict(entities)


def extract_tagged_abstract(abstract_node):
    """
    Extract the string between the abstract tags
    :param abstract_node: An Element representing the abstract tag
    :return: A string containing the text between the abstract tags
    """
    abstract_string = ET.tostring(abstract_node).decode('utf-8')
    return re.search(abstract_regex, abstract_string).group(1).strip()


def parse_abstract_text(id, title, abstract_node):
    """
    Generate a data frame representing an abstract given the function's parameters. Code is written based on trial data
    for subtask 1.1
    :param id: A string representing the id attribute of the text tag
    :param title: A string representing the title of the abstract
    :param abstract_node: An Element (from Element Tree) representing the abstract tag
    :return: A dataframe containing the text's id, abstract's title, sentence number, tagged and untagged sentence,
            and a dictionary of entities in that sentence
    """
    column_order = ['text_id', 'title', 'sent_num', 'tagged_sentence', 'untagged_sentence', 'entity_dict']
    abstract_sents_tagged = sent_detector.tokenize(extract_tagged_abstract(abstract_node))
    abstract_sents_untagged = sent_detector.tokenize(ET.tostring(abstract_node, method='text').decode('utf-8'))
    entities_per_sent = [extract_entities(sent) for sent in abstract_sents_tagged]
    df = pd.DataFrame({'tagged_sentence': abstract_sents_tagged,
                       'untagged_sentence': abstract_sents_untagged,
                       'entity_dict': entities_per_sent,
                       'text_id': id,
                       'title': title,
                       'sent_num': range(1, len(abstract_sents_tagged) + 1)})
    df = df[column_order]
    df = df.set_index(['text_id', 'sent_num'])
    return df


def generate_abstract_dataframe(abstractfilepath):
    """
    Generate a dataframe representing the abstracts stored in the given xml file
    :param abstractfilepath: A string representing the path to the xml file containing the abstracts
    :return: A data frame that is the concatenation of the call to parse_abstract_text() on each text tag in the xml
             file
    """
    tree = ET.parse(abstractfilepath)
    root = tree.getroot()
    df = pd.concat([parse_abstract_text(text.get('id'),
                                        text.find('title').text,
                                        text.find('abstract')) for text in root.getchildren()], axis=0)

    return df


def get_relation_id(row, ref_series):
    """
    Look up the text id and sentence number for a given relation using ref_df. Look up is done based on the given pair
    of entity ids and assuming that a given relation (based on entity ids) can be found only once in the corpus.
    :param row: A two column data frame containing the entity ids for entity 1 and entity 2 in a given relation.
    :param ref_series: A pandas Series object where each row contains the  dictionary containing the
                  entities in that sentence and is indexed by text id and sentence number.
    :return: A 1 x 2 data frame where column 1 is the text id and column 2 is the sentence number for the given row.
    """
    eid1, eid2 = row
    idx = ref_series.apply(lambda dict, id1, id2: (id1 in dict) & (id2 in dict), id1=eid1, id2=eid2)
    # check that the pair of entity ids occur only once in the data set!
    assert (sum(idx) == 1), 'The entity id pair ({}, {}) occured {} times in the dataset!'.format(eid1, eid2, sum(idx))
    # res = ref_series[['text_id', 'sent_num']].where(idx).dropna()
    res = list(ref_series[idx].index[0])
    return res


def generate_relations_dataframe(relationfilepath, abstract_df):
    """
    Generate a data frame containing the relations in the text file in relationfilepath.
    :param relationfilepath: A string representing the path to the txt file listing the relations.
    :param abstract_df: A data frame generated from the call to generate_abstract_dataframe() which will be used to look
                        up each relation's text id and sentence number.
    :return: A data frame with columns text_id, sent_num, relation, entity1, entity2, is_reversed.
    """
    relations_df = pd.read_table(relationfilepath, names=['tmp'])
    relations_df = relations_df['tmp'].str.extract(relation_regex, expand=True)

    reversed_relations = relations_df['is_reversed'] == 'REVERSE'
    relations_df['is_reversed'] = np.where(reversed_relations, 1, 0)

    ref = abstract_df['entity_dict']

    relation_ids = relations_df[['entity1', 'entity2']].apply(get_relation_id, axis=1, ref_series=ref)

    relations_df[['text_id', 'sent_num']] = relation_ids
    relations_df = relations_df.set_index(['text_id', 'sent_num'])

    return relations_df
