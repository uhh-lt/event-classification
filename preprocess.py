"""
Preprocess raw documents to perform event classification on.
"""
from typing import Optional
import os
import json

import catma_gitlab as catma
import typer
import spacy
from spacy.tokens import Doc, Token
from thinc.api import set_gpu_allocator, require_gpu

from event_classify.datasets import SimpleEventDataset
import event_classify.segmentations # noqa: W0611

app = typer.Typer()


def build_pipeline() -> spacy.Language:
    nlp = spacy.load("de_dep_news_trf")
    nlp.add_pipe("event_segmentation", after="parser")
    return nlp


@app.command()
def preprocess(text_file_path: str, out_file_path: str, title: Optional[str] = None, gpu: bool = False):
    """
    Segment document into event spans based on verb occurences.

    Creates a JSON file with the document and its event spans.
    """
    if gpu:
        set_gpu_allocator("pytorch")
        require_gpu(0)

    in_file = open(text_file_path, "r")
    full_text = "\n".join(in_file.readlines())
    nlp = build_pipeline()
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
    json.dump(data, open(out_file_path, "w"))


@app.command()
def spans(input_sentence: str, display: bool = False):
    nlp = build_pipeline()
    doc = nlp(input_sentence)
    if display:
        spacy.displacy.serve(doc, style="dep")
    for token in doc:
        print(token, token.tag_, token.pos_, token.dep_)
    for ranges in doc._.events:
        print("======")
        for event_range in ranges:
            print(event_range)


@app.command()
def eval():
    """
    Segment document into event spans based on verb occurences.

    Creates a JSON file with the document and its event spans.
    """
    # set_gpu_allocator("pytorch")
    # require_gpu(0)
    gold_external_spans = set()
    verwandlung_dataset, text = get_verwandlung()
    for annotation in verwandlung_dataset:
        gold_external_spans.add((annotation.start, annotation.end))
    nlp = build_pipeline()
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
    for p, g in zip(predict[:40], gold[:40]):
        print("======")
        print(g, p)
        print("GOLD:")
        print(text[g[0]:g[1]])
        print("Predict:")
        print(text[p[0]:p[1]])
    print("Num Predicted:", len(predict_external_spans))
    print("Num Gold", len(gold_external_spans))
    print("Accuracy:", accuracy)
    import ipdb; ipdb.set_trace()


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
