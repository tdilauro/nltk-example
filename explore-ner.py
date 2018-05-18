import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import json
import re
import sys

filename = 'articles.json'
specific_record = None
if len(sys.argv) > 1:
  filename = sys.argv[1]
if len(sys.argv) > 2:
  specific_record = sys.argv[2]

def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

def load_articles():
  document_contents = []
  doc_ids = []

  with open(filename) as json_data:
    jd = json.load(json_data)
    for row in jd:
      doc_ids.append(row['identifier'])
      document_contents.append(row['content'])
  return document_contents, doc_ids

docs, doc_ids = load_articles()

for i,doc in enumerate(docs):
  sentences = sent_tokenize(doc)
  for j,s in enumerate(sentences):
    toks = word_tokenize(s)
    print " >> " + s
    #for t in toks: 
    #  print "      - " + t
    sentences[j] = toks
  tagged = nltk.pos_tag_sents(sentences)
  #print tagged
  named_entities = nltk.ne_chunk_sents(tagged)

  print " ** NLTK NER ** " 
  ner = {}
  for stree in named_entities:
    for subtree in stree.subtrees(filter=lambda t: len(t.label()) > 0):
      type = subtree.label()
      if type != 'S': 
        name = " ".join(c[0] for c in subtree.leaves())
        #print str(type) + ": " + name
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
    print "   ", ner[w], w, " TYPE: ", t, "  ", v
    if t == "LOCATION":
      mylocation = v
    if mylocation == '' and topgspgpe == '' and (t == 'GSP' or t == 'GPE'):
      topgspgpe = v
  if mylocation == '':
    mylocation = topgspgpe
    if mylocation == '':
      mylocation = 'undetected'
  print "LOCATION: ", mylocation
