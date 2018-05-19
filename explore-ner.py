from __future__ import print_function

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import argparse
import ijson
import re


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
        sentences = sent_tokenize(content)
        for j,s in enumerate(sentences):
            toks = word_tokenize(s)
            print(" >> " + s)
            #for t in toks:
            #  print("      - " + t)
            sentences[j] = toks
        tagged = nltk.pos_tag_sents(sentences)
        #print(tagged)
        named_entities = nltk.ne_chunk_sents(tagged)

        print(" ** NLTK NER ** ")
        ner = {}
        for stree in named_entities:
            for subtree in stree.subtrees(filter=lambda t: len(t.label()) > 0):
                type = subtree.label()
                if type != 'S':
                    name = " ".join(c[0] for c in subtree.leaves())
                    #print(str(type) + ": " + name)
                    try:
                        ner[ str(type) + ": " + name ] += 1
                    except:
                        ner[ str(type) + ": " + name ] = 1
        mylocation = ''
        topgspgpe = ''
        for w in sorted(ner, key=ner.get, reverse=True):
            if ner[w] < 2:
                break
            m = re.search('^(\w+):\s+(.+)', w)
            if m is not None:
                t = m.group(1)
                v = m.group(2)
            print("   ", ner[w], w, " TYPE: ", t, "  ", v)
            if t == "LOCATION":
                mylocation = v
            if mylocation == '' and topgspgpe == '' and (t == 'GSP' or t == 'GPE'):
                topgspgpe = v
        if mylocation == '':
            mylocation = topgspgpe
            if mylocation == '':
                mylocation = 'undetected'
        print("LOCATION: ", mylocation)


def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)


def json_objects_from_file(filename):
    with open(filename, 'rb') as f:
        for obj in ijson.items(f, 'item'):
            yield obj


if __name__ == '__main__':
    main()
