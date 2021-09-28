import gc
import json
import math
import os
from typing import NamedTuple, List

import catma_gitlab as catma
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import typer

from event_classify.datasets import (
    EventType,
    JSONDataset,
    SimpleEventDataset,
    SpanAnnotation,
)
from event_classify.eval import evaluate
from event_classify.parser import Parser
import event_classify.preprocessing
from event_classify.preprocessing import build_pipeline
from event_classify.util import get_model

app = typer.Typer()

@app.command()
def main(segmented_json: str, out_path: str, model_path: str, device: str = "cuda:0", batch_size: int = 16, special_tokens: bool = True):
    """
    Add predictions to segmented json file.

    Args:
    model_path: Path to run directory containing saved model and tokenizer
    """
    dataset = JSONDataset(segmented_json, include_special_tokens=special_tokens)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer),
    )
    model, tokenizer = get_model(model_path)
    model.to(device)
    _, _, predictions = evaluate(loader, model, device=device)
    dataset.save_json(out_path, [EventType(p.item()) for p in predictions])


@app.command()
def gold_spans(model_path: str, device: str = "cuda:0", special_tokens: bool = True, dev: bool = False):
    project = catma.CatmaProject(
        ".",
        "CATMA_DD5E9DF1-0F5C-4FBD-B333-D507976CA3C7_EvENT_root",
        filter_intrinsic_markup=False,
    )
    in_distribution_dataset = SimpleEventDataset(
        project,
        ["Effi_Briest_MW", "Krambambuli_MW"],
        include_special_tokens=special_tokens,
    )
    train_size = math.floor(len(in_distribution_dataset) * 0.9)
    dev_size = len(in_distribution_dataset) - train_size
    train_dataset, dev_dataset = random_split(
        in_distribution_dataset,
        [train_size, dev_size],
        generator=torch.Generator().manual_seed(13),
    )
    collection = project.ac_dict["Verwandlung_MV"]
    test_dataset = SimpleEventDataset(
        project,
        ["Verwandlung_MV"],
        include_special_tokens=special_tokens,
    )
    model, tokenizer = get_model(model_path)
    model.eval()
    print(len(dev_dataset))
    loader = DataLoader(
        dev_dataset if dev else test_dataset,
        batch_size=16,
        collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer),
    )
    evaluate(
        loader,
        model,
        device=device,
        out_file=open("gold_spans.json", "w"),
        save_confusion_matrix=True,
    )


@app.command()
def dprose(model_path: str, output_name: str, device: str = "cuda:0", special_tokens: bool = True, batch_size: int = 8):
    """
    Predict all of d-prose, applying segmentation in the same step.
    """
    if device.startswith("cuda"):
        event_classify.preprocessing.use_gpu()
    out_file = open(output_name, "w")
    ids = open("d-prose/d-prose_ids.csv")
    # skip header
    _ = next(ids)
    nlp = build_pipeline(Parser.SPACY)
    for dprose_id, name in tqdm([line.split(",") for line in ids]):
        # We want to throw out the previous iterations memory
        # doc = None
        # gc.collect()
        # This way we might not run out of memory after a few docs...
        dprose_id = int(dprose_id)
        name = name.strip()
        in_file = open(os.path.join("d-prose", name + ".txt"))
        full_text = "".join(in_file.readlines())
        splits = split_text(full_text)
        # Sanity check, splitting should change text!
        assert full_text == "".join(split.text for split in splits)
        data = {
            "text": full_text,
            "title": name,
            "annotations": []
        }
        for split in splits:
            doc = nlp(split.text)
            annotations = event_classify.preprocessing.get_annotation_dicts(doc)
            for annotation in annotations:
                annotation["start"] += split.offset
                annotation["end"] += split.offset
                new_spans = []
                for span in annotation["spans"]:
                    new_spans.append((
                        span[0] + split.offset,
                        span[1] + split.offset,
                    ))
                annotation["spans"] = new_spans
            data["annotations"].extend(annotations)
        towards_end = data["annotations"][-10]
        print(full_text[towards_end["start"]:towards_end["end"]])
        dataset = JSONDataset(dataset_file=None, data=[data], include_special_tokens=special_tokens)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer),
        )
        model, tokenizer = get_model(model_path)
        model.to(device)
        _, _, predictions = evaluate(loader, model, device=device)
        # We only pass in one document, so we only use [0]
        data = dataset.get_annotation_json([EventType(p.item()) for p in predictions])[0]
        data["dprose_id"] = dprose_id
        json.dump(data, out_file)
        out_file.flush()
        out_file.write("\n")
        out_file.flush()


@app.command()
def plain_text_file(model_path: str, in_name: str, output_name: str, device: str = "cuda:0", special_tokens: bool = True, batch_size: int = 8):
    """
    Predict a plain text file.
    """
    if device.startswith("cuda"):
        event_classify.preprocessing.use_gpu()
    out_file = open(output_name, "w")
    nlp = build_pipeline(Parser.SPACY)
    in_file = open(in_name, "r")
    full_text = "".join(in_file.readlines())
    splits = split_text(full_text)
    # Sanity check, splitting should change text!
    assert full_text == "".join(split.text for split in splits)
    data = {
        "text": full_text,
        "title": os.path.basename(in_name),
        "annotations": []
    }
    for split in splits:
        doc = nlp(split.text)
        annotations = event_classify.preprocessing.get_annotation_dicts(doc)
        for annotation in annotations:
            annotation["start"] += split.offset
            annotation["end"] += split.offset
            new_spans = []
            for span in annotation["spans"]:
                new_spans.append((
                    span[0] + split.offset,
                    span[1] + split.offset,
                ))
            annotation["spans"] = new_spans
        data["annotations"].extend(annotations)
    towards_end = data["annotations"][-10]
    print(full_text[towards_end["start"]:towards_end["end"]])
    dataset = JSONDataset(dataset_file=None, data=[data], include_special_tokens=special_tokens)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer),
    )
    model, tokenizer = get_model(model_path)
    model.to(device)
    _, _, predictions = evaluate(loader, model, device=device)
    # We only pass in one document, so we only use [0]
    data = dataset.get_annotation_json([EventType(p.item()) for p in predictions])[0]
    json.dump(data, out_file)
    out_file.flush()
    out_file.write("\n")
    out_file.flush()

class SubDoc(NamedTuple):
    offset: int
    text: str


def split_text(text: str, allowed_split=".\n") -> List[SubDoc]:
    """
    Split text into a number of sub strings managable for spacy.

    This could fail for abritrary sequences but the books in d-prose all have
    seem to have '.\n'
    """
    total = len(text)
    # This is super conservative spacy seems to be able to do >300k tokens on 12GB VRAM
    max_segment_length = 100000
    segments = text.split(allowed_split)
    out = []
    current_split = []
    for i, segment in enumerate(segments):
        if len(segment) > max_segment_length:
            if allowed_split != ". ":
                # Try again with the double newline strategy
                return split_text(text, allowed_split=". ")
            else:
                raise ValueError("Document has too few split options.")
        if (sum(len(s) for s in current_split) + len(segment)) <= max_segment_length:
            if i == len(segments) - 1:
                current_split.append(segment)
            else:
                current_split.append(segment + allowed_split)
        else:
            out.append(SubDoc(text="".join(current_split), offset=sum(len(split.text) for split in out)))
            if i == len(segments) - 1:
                current_split = [segment]
            else:
                current_split = [segment + allowed_split]
    out.append(SubDoc(text="".join(current_split), offset=sum(len(split.text) for split in out)))
    return out


if __name__ == "__main__":
    app()
