from __future__ import print_function

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import argparse
import ijson


DEFAULT_FILENAME = 'articles.json'
DEFAULT_RECORD = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', default=DEFAULT_FILENAME, nargs='?', help='echo the string you use here')
    parser.add_argument('record', default=DEFAULT_RECORD, nargs='?', help='record number (all, if omitted)')
    args = parser.parse_args()

    filename = args.filename
    specific_record = args.record

    documents = json_objects_from_file(filename)

    for i, doc in enumerate(documents):
        content = doc['content']
        do_ner(content)


def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)


NLTK_TREE_SENTENCE_LABEL = 'S'

def do_ner(content):
    sentences = sent_tokenize(content)
    tokenized_sentences = [word_tokenize(s) for s in sentences]
    tagged_sentences = nltk.pos_tag_sents(tokenized_sentences)
    named_entities = nltk.ne_chunk_sents(tagged_sentences)

    labeled_entities = {}
    for stree in named_entities:
        for subtree in stree.subtrees(filter=lambda t: len(t.label()) > 0):
            node_type = subtree.label()
            if node_type != NLTK_TREE_SENTENCE_LABEL:
                name = " ".join(c[0] for c in subtree.leaves())
                # print(str(type) + ": " + name)
                entity_key = (node_type, name)
                try:
                    labeled_entities[entity_key] += 1
                except:
                    labeled_entities[entity_key] = 1

    mylocation = ''
    topgspgpe = ''
    for w in sorted(labeled_entities, key=labeled_entities.get, reverse=True):
        if labeled_entities[w] < 2:
            break
        t, v = w
        print("   ", labeled_entities[w], w, " TYPE: ", t, "  ", v)
        if t == "LOCATION":
            mylocation = v
        if mylocation == '' and topgspgpe == '' and (t == 'GSP' or t == 'GPE'):
            topgspgpe = v
    if mylocation == '':
        mylocation = topgspgpe
        if mylocation == '':
            mylocation = 'undetected'
    print("LOCATION: ", mylocation)


def json_objects_from_file(filename):
    with open(filename, 'rb') as f:
        for obj in ijson.items(f, 'item'):
            yield obj


if __name__ == '__main__':
    main()
