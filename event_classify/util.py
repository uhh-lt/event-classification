from transformers import ElectraTokenizer, ElectraForSequenceClassification
from .datasets import EventType
import os
from typing import List


def get_model(model_path: str):
    tokenizer: ElectraTokenizer = ElectraTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer")
    )
    model = ElectraForSequenceClassification.from_pretrained(
        os.path.join(model_path, "best-model"),
        num_labels=4,
    )
    return model, tokenizer


def smooth_bins(annotation_spans: List, smooth_character_span=1000) -> List[int]:
    """
    Smooth narrativity score for all event in a text segment.

    This function is based on Michael Vauth's code.
    """
    smooth_narrativity_scores = []
    start_points = []

    max_end = max(span["end"] for span in annotation_spans)
    for b in range(int(smooth_character_span / 2), max_end, 100): # iterates in 100 character steps over the text
        start_points.append(b)
        annotation_spans_filtered = []
        for anno_span in annotation_spans:
            if (anno_span["start"] > b - smooth_character_span) \
                    and (anno_span["end"] < b + smooth_character_span):
                annotation_spans_filtered.append(anno_span)

        smooth_narrativity_scores.append(
            sum(EventType.from_tag_name(span["predicted"]).get_narrativity_score() for span in annotation_spans_filtered)
        )
    return smooth_narrativity_scores
