import xml.etree.ElementTree as ET
from collections import OrderedDict
import nltk.data
import re
import pandas as pd

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
entity_regex = re.compile(r'<entity id=\"([A-Z][\d.-]+)\">(.*?)</entity>')
abstract_regex = re.compile(r'<abstract>((?:.|\s)*)</abstract>')


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


def generate_abstract_dataframe(id, title, abstract_node):
    """
    Generate a data frame for each text tag in the training/test data. Code is written based on trial data for
    subtask 1.1
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
    return df[column_order]
