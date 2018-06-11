from __future__ import print_function
from __future__ import unicode_literals

import collections
import spacy


class NER(object):
    def __init__(self, model=None, **kwargs):
        self.model = model
        if model is not None:
            self.nlp = spacy.load(model)
        else:
            self.nlp = None

    def document(self, **kwargs):
        return Document(_ner=self, **kwargs)


class Document(object):
    def __init__(self, content=None, content_type=None, doc_id=None, metadata=None, heuristic_min_count=2,
                 _ner=None, **kwargs):
        self.id = doc_id
        self.content = content
        self.metadata = metadata
        self.content_type = content_type

        doc = _ner.nlp(content)
        self.sents = doc.sents
        self.ents = doc.ents
        self.sentences = [sent.string for sent in doc.sents]

        self.typed_ent_counter = typed_entity_counts(doc.ents)


def typed_entity_counts(named_entities):
    # labeled_entities = {}
    counter = collections.Counter()
    for ent in named_entities:
        counter.update([(ent.label_, ent.string.strip())])
    return counter
