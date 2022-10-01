import re
from spacy.lang.en import English

nlpSeg = English()
nlpSeg.add_pipe("sentencizer")

def process_stops(text):
    processed_text = text
    version = re.compile(r'(\d+)\.(\d+)\.(\d+)')
    decimal = re.compile(r'(\d+)\.(\d+)')

    processed_text = re.sub(version, lambda matchobj: "_".join(list(matchobj.groups())), processed_text)
    processed_text = re.sub(decimal, lambda matchobj: "_".join(list(matchobj.groups())), processed_text)
    return processed_text

def reverse_process_stops(text):
    processed_text = text
    version = re.compile(r'(\d+)\_(\d+)\_(\d+)')
    decimal = re.compile(r'(\d+)\_(\d+)')

    processed_text = re.sub(version, lambda matchobj: ".".join(list(matchobj.groups())), processed_text)
    processed_text = re.sub(decimal, lambda matchobj: ".".join(list(matchobj.groups())), processed_text)
    return processed_text

# def sentence_segmentation(text):
#     doc = nlpSeg(text)
#     for sent in doc.sents:
