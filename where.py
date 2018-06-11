from __future__ import print_function
from __future__ import unicode_literals

import argparse
import ijson
import os


DEFAULT_FILENAME = 'articles.json'
DEFAULT_RECORD = None

# cd ~/Dropbox/data/models/spaCy
# wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.0.0/en_core_web_lg-2.0.0.tar.gz
# tar xfz en_core_web_lg-2.0.0.tar.gz
SPACY_MODEL = '~/Dropbox/data/models/spaCy/en_core_web_lg-2.0.0/en_core_web_lg/en_core_web_lg-2.0.0'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--toolkit', '-t', choices=set(('spacy', 'nltk')), help='toolkit to use')
    parser.add_argument('filename', default=DEFAULT_FILENAME, nargs='?', help='echo the string you use here')
    parser.add_argument('record', default=DEFAULT_RECORD, nargs='?', type=int, help='record number (all, if omitted)')
    args = parser.parse_args()

    if args.toolkit is None or args.toolkit == 'spacy':
        from ner_spacy import NER
        ner = NER(model=os.path.expanduser(SPACY_MODEL))
    elif args.toolkit == 'nltk':
        from ner_nltk import NER
        # import nltk
        # ner = NER(nltk.ne_chunk_sents)
        ner = NER()
    else:
        raise RuntimeError('invalid NL toolkit: "{}"'.format(args.toolkit))

    filename = args.filename
    specific_record = args.record
    if specific_record is not None:
        specific_record = [specific_record]

    items = json_objects_from_file(filename)

    for result in process_items(items, ner=ner, records=specific_record):
        doc = result['document']
        doc_properties = apply_heuristics(doc.ents, typed_entity_counter=doc.typed_ent_counter, min_count=2)
        print("*** Document {} ***".format(doc.id))
        print(doc.content.encode('utf-8'))
        counter = doc.typed_ent_counter
        for k in counter.keys():
            if counter[k] < 2:
                continue
            node_type, phrase = k
            print('   {} {}  TYPE: {} {}'.format(counter[k], k, node_type, phrase))
        print('LOCATION: {location} ({topgspgpe})'.format(**doc_properties))


def apply_heuristics(named_entities, typed_entity_counter=None, min_count=0):
    mylocation = ''
    topgspgpe = ''
    for w in sorted(typed_entity_counter, key=typed_entity_counter.get, reverse=True):
        if typed_entity_counter[w] < min_count:
            continue
        node_type, phrase = w
        if node_type in ['LOCATION', 'LOC']:
            mylocation = phrase
        if mylocation == '' and topgspgpe == '' and (node_type in ['NORP', 'GSP', 'GPE']):
            topgspgpe = phrase
    if mylocation == '':
        mylocation = topgspgpe
        if mylocation == '':
            mylocation = 'undetected'
    return {'location': mylocation, 'topgspgpe': topgspgpe}


def process_items(items, ner=None, records=None):
    for i, item in enumerate(items):
        if records is not None and item['identifier'] not in records:
            continue
        doc = ner.document(content=item['content'], doc_id=item['identifier'])
        yield {'seq': i, 'document': doc}


def json_objects_from_file(filename):
    with open(filename, 'rb') as f:
        for obj in ijson.items(f, 'item'):
            yield obj


if __name__ == '__main__':
    main()
