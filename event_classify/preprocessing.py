import spacy
from spacy.tokens import Doc, Token
from thinc.api import require_gpu, set_gpu_allocator

from event_classify.parser import HermaParser, ParZuParser, Parser
import event_classify.segmentations


def build_pipeline(parser: Parser) -> spacy.Language:
    """
    Builds a spacy pipeline with the parser of your choice
    """
    if parser == Parser.SPACY:
        nlp = spacy.load("de_dep_news_trf")
        nlp.add_pipe("event_segmentation", after="parser")
    elif parser == Parser.PARZU:
        nlp = spacy.load("de_dep_news_trf", disable=["parser"])
        nlp.add_pipe("sentencizer")
        nlp.add_pipe("parzu_parser")
        nlp.add_pipe("event_segmentation", after="parzu_parser")
    elif parser == Parser.HERMA:
        nlp = spacy.load("de_dep_news_trf", disable=["parser"])
        nlp.add_pipe("sentencizer")
        nlp.add_pipe("herma_parser")
        nlp.add_pipe("event_segmentation", after="herma_parser")
    return nlp


def get_annotation_dicts(doc: Doc):
    annotations = []
    for event_ranges in doc._.events:
        spans = []
        for subspan in event_ranges:
            spans.append((subspan.start_char, subspan.end_char))
        annotations.append({
            "start": min([start for start, end in spans]),
            "end": max([end for start, end in spans]),
            "spans": spans,
            "predicted": None,
        })
    return annotations


def use_gpu():
    set_gpu_allocator("pytorch")
    require_gpu(0)
