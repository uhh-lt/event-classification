from transformers import ElectraTokenizer, ElectraForSequenceClassification
from .datasets import EventType
import os
from typing import List
import bisect


def get_model(model_path: str):
    tokenizer: ElectraTokenizer = ElectraTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer")
    )
    model = ElectraForSequenceClassification.from_pretrained(
        os.path.join(model_path, "best-model"),
        num_labels=4,
    )
    return model, tokenizer


def filter_sorted(data, start_value, end_value, key=lambda x: x, key_start=None, key_end=None):
    if key_start is None:
        key_start = key
    if key_end is None:
        key_end = key
    start_index = bisect.bisect_left([key_start(el) for el in data], start_value)
    end_index = bisect.bisect_right([key_end(el) for el in data], end_value)
    return data[start_index:end_index]


def smooth_bins(annotation_spans: List, smooth_character_span=1000) -> List[int]:
    """
    Smooth narrativity score for all event in a text segment.

    This function is based on Michael Vauth's code.
    """
    smooth_narrativity_scores = []
    start_points = []

    max_end = max(span["end"] for span in annotation_spans)
    starts = [span["start"] for span in annotation_spans]
    ends = [span["end"] for span in annotation_spans]
    for b in range(
        int(smooth_character_span / 2), max_end, 100
    ):  # iterates in 100 character steps over the text
        start_points.append(b)
        start_index = bisect.bisect_left(starts, b - smooth_character_span)
        end_index = bisect.bisect_right(ends, b + smooth_character_span)
        annotation_spans_filtered = annotation_spans[start_index:end_index - 10]
        # The ends are not necessarily sorted (sorted by start), so let's check around here if there is anything we need to fix
        for anno_span in annotation_spans[end_index - 10: end_index + 10]:
            if (anno_span["start"] > b - smooth_character_span) and (
                anno_span["end"] < b + smooth_character_span
            ):
                annotation_spans_filtered.append(anno_span)

        smooth_narrativity_scores.append(
            sum(
                EventType.from_tag_name(span["predicted"]).get_narrativity_score()
                for span in annotation_spans_filtered
            )
        )
    return smooth_narrativity_scores
