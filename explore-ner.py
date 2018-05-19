from __future__ import print_function
from __future__ import unicode_literals

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import argparse
import collections
import ijson


DEFAULT_FILENAME = 'articles.json'
DEFAULT_RECORD = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', default=DEFAULT_FILENAME, nargs='?', help='echo the string you use here')
    parser.add_argument('record', default=DEFAULT_RECORD, nargs='?', type=int, help='record number (all, if omitted)')
    args = parser.parse_args()

    filename = args.filename
    specific_record = args.record
    if specific_record is not None:
        specific_record = [specific_record]

    documents = json_objects_from_file(filename)

    for result in process_documents(documents, records=specific_record):
        doc = result['document']
        print("*** Document {} ***".format(doc['identifier']))
        print(doc['content'].encode('utf-8'))
        counter = result['heuristics']['typed_entity_counter']
        for k in counter.keys():
            if counter[k] < 2:
                continue
            node_type, phrase = k
            print('   {} {}  TYPE: {} {}'.format(counter[k], k, node_type, phrase))
        print('LOCATION: {} ({})'.format(result['heuristics']['location'], result['heuristics']['topgspgpe']))


def process_documents(documents, records=None):
    for i, doc in enumerate(documents):
        if records is not None and doc['identifier'] not in records:
            continue
        content = doc['content']
        ner_results = do_ner(content)
        heur_results = apply_heuristics(ner_results['named_entities'], min_count=2)
        yield {'seq': i, 'document': doc, 'ner': ner_results, 'heuristics': heur_results}


def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)


NLTK_TREE_SENTENCE_LABEL = 'S'

def do_ner(content):
    sentences = sent_tokenize(content)
    tokenized_sentences = [word_tokenize(s) for s in sentences]
    tagged_sentences = nltk.pos_tag_sents(tokenized_sentences)
    named_entities = nltk.ne_chunk_sents(tagged_sentences)
    return {'named_entities': named_entities,
            'sentences': sentences,
            'tokenized_sentences': tokenized_sentences,
            'tagged_sentences': tagged_sentences}


subtree_filter = lambda st: len(st.label()) > 0 and st.label() != NLTK_TREE_SENTENCE_LABEL
def typed_entity_counts(named_entities):
    # labeled_entities = {}
    counter = collections.Counter()
    # only non-sentence nodes with non-empty labels
    for stree in named_entities:
        for subtree in stree.subtrees(filter=subtree_filter):
            phrase = " ".join(c[0] for c in subtree.leaves())
            counter.update([(subtree.label(), phrase)])
    return counter


def apply_heuristics(named_entities, min_count=0):
    typed_entity_counter = typed_entity_counts(named_entities)

    mylocation = ''
    topgspgpe = ''
    for w in sorted(typed_entity_counter, key=typed_entity_counter.get, reverse=True):
        if typed_entity_counter[w] < min_count:
            continue
        node_type, phrase = w
        if node_type == "LOCATION":
            mylocation = phrase
        if mylocation == '' and topgspgpe == '' and (node_type == 'GSP' or node_type == 'GPE'):
            topgspgpe = phrase
    if mylocation == '':
        mylocation = topgspgpe
        if mylocation == '':
            mylocation = 'undetected'
    return {'location': mylocation, 'topgspgpe': topgspgpe, 'typed_entity_counter': typed_entity_counter}


def json_objects_from_file(filename):
    with open(filename, 'rb') as f:
        for obj in ijson.items(f, 'item'):
            yield obj


if __name__ == '__main__':
    main()
