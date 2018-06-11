from __future__ import unicode_literals

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import collections


class NER(object):
    def __init__(self, chunker=None):
        if chunker is None:
            chunker = nltk.ne_chunk_sents
        self.chunker = chunker

    def document(self, **kwargs):
        return Document(_ner=self, **kwargs)


class Document (object):
    def __init__(self, content=None, content_type=None, doc_id=None, metadata=None, heuristic_min_count=2,
                 _ner=None, **kwargs):
        self.id = doc_id
        self.content = content
        self.metadata = metadata
        self.content_type = content_type

        ner_results = do_ner(content, chunker=_ner.chunker)
        self.ents = ner_results['named_entities']
        self.sents = ner_results['sentences']
        self.tokenized = ner_results['tokenized_sentences']
        self.pos = ner_results['tagged_sentences']

        self.typed_ent_counter = typed_entity_counts(self.ents)


NLTK_TREE_SENTENCE_LABEL = 'S'

def do_ner(content, chunker=None):
    sentences = sent_tokenize(content)
    tokenized_sentences = [word_tokenize(s) for s in sentences]
    tagged_sentences = nltk.pos_tag_sents(tokenized_sentences)
    named_entities = chunker(tagged_sentences)
    return {'named_entities': named_entities,
            'sentences': sentences,
            'tokenized_sentences': tokenized_sentences,
            'tagged_sentences': tagged_sentences}


# only non-sentence nodes with non-empty labels
subtree_filter = lambda st: len(st.label()) > 0 and st.label() != NLTK_TREE_SENTENCE_LABEL

def typed_entity_counts(named_entities):
    # labeled_entities = {}
    counter = collections.Counter()
    for stree in named_entities:
        for subtree in stree.subtrees(filter=subtree_filter):
            phrase = " ".join(c[0] for c in subtree.leaves())
            counter.update([(subtree.label(), phrase)])
    return counter
