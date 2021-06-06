"""
Preprocess raw documents to perform event classification on.
"""
from typing import Optional
import os
import json
import bisect
from typing import List

import catma_gitlab as catma
import typer
import spacy
from spacy.tokens import Doc, Token
from thinc.api import set_gpu_allocator, require_gpu

from event_classify.datasets import SimpleEventDataset
import event_classify.segmentations # noqa: W0611
from event_classify.parser import HermaParser, ParZuParser, Parser

app = typer.Typer()


def build_pipeline(parser: Parser) -> spacy.Language:
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


@app.command()
def preprocess(text_file_paths: List[str], out_file_path: str, title: Optional[str] = None, gpu: bool = False, parser: Parser = typer.Option(Parser.SPACY)):
    """
    Segment a set document into event spans based on verb occurrences.

    Creates a JSON file with the document and its event spans suitable for passing to the `predict.py`.
    """
    if gpu:
        set_gpu_allocator("pytorch")
        require_gpu(0)

    nlp = build_pipeline(parser)
    document_list = []
    for text_file_path in text_file_paths:
        in_file = open(text_file_path, "r")
        full_text = "\n".join(in_file.readlines())
        doc = nlp(full_text)
        if title is None:
            title, _ = os.path.splitext(os.path.basename(out_file_path))
        data = {
            "text": full_text,
            "title": title,
            "annotations": []
        }
        for event_ranges in doc._.events:
            spans = []
            for subspan in event_ranges:
                spans.append((subspan.start_char, subspan.end_char))
            data["annotations"].append({
                "start": min([start for start, end in spans]),
                "end": max([end for start, end in spans]),
                "spans": spans,
                "predicted": None,
            })
        document_list.append(data)
    json.dump(document_list, open(out_file_path, "w"))


@app.command()
def spans(input_sentence: str, display: bool = False, parser: Parser = typer.Option(Parser.SPACY)):
    nlp = build_pipeline(parser)
    doc = nlp(input_sentence)
    if display:
        spacy.displacy.serve(doc, style="dep")
    for token in doc:
        print(token, token.tag_, token.pos_, token.dep_, token._.custom_dep)
    for ranges in doc._.events:
        print("======")
        for event_range in ranges:
            print(event_range)


@app.command()
def eval(parser: Parser = typer.Option(Parser.SPACY)):
    """
    Evaluate segmentation outputs.
    """
    # set_gpu_allocator("pytorch")
    # require_gpu(0)
    gold_external_spans = set()
    verwandlung_dataset, text = get_verwandlung()
    for annotation in verwandlung_dataset:
        gold_external_spans.add((annotation.start, annotation.end))
    nlp = build_pipeline(parser)
    doc = nlp(text)
    predict_external_spans = set()
    for event_ranges in doc._.events:
        start = min(r.start_char for r in event_ranges)
        end = max(r.end_char for r in event_ranges)
        predict_external_spans.add((start, end))
    overlap = len(predict_external_spans & gold_external_spans)
    accuracy =  overlap / max([len(predict_external_spans), len(gold_external_spans)])
    predict = sorted(predict_external_spans)
    gold = sorted(gold_external_spans)
    for gold_span in gold[:100]:
        print("===========")
        pred_span = predict[find_best_match_for_start(predict, gold_span)]
        print("GOLD:")
        print(text[gold_span[0]:gold_span[1]])
        print("Predict:")
        print(text[pred_span[0]:pred_span[1]])
    print("Num Predicted:", len(predict_external_spans))
    print("Num Gold", len(gold_external_spans))
    print("Accuracy:", accuracy)


def find_best_match_for_start(sorted_list, element):
    i = bisect.bisect_left(sorted_list, element)
    if i != len(sorted_list):
        return i
    raise ValueError


def get_verwandlung():
    project = catma.CatmaProject(
        ".",
        "CATMA_DD5E9DF1-0F5C-4FBD-B333-D507976CA3C7_EvENT_root",
        filter_intrinsic_markup=False,
    )
    collection = project.ac_dict["Verwandlung_MV"]
    dataset = SimpleEventDataset(
        project,
        ["Verwandlung_MV"],
    )
    return dataset, collection.text.plain_text

if __name__ == "__main__":
    app()
