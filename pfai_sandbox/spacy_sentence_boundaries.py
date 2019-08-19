import spacy

nlp = spacy.load('en_core_web_sm')
S = nlp(u'Spacy is a useful tool. Can it can help extract sentences? Yes, it can!')
for s in S.sents:
    print(s)