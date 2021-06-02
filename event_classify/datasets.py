from __future__ import annotations
from typing import NamedTuple, Iterable, List
from enum import Enum
import torch
from torch.utils.data import Dataset
import catma_gitlab as catma
from transformers import PreTrainedTokenizer


class EventType(Enum):
    NON_EVENT = 0
    CHANGE_OF_STATE = 1
    PROCESS = 2
    STATIVE_EVENT = 3

    def to_onehot(self):
        out = torch.zeros(4)
        out[self.value] = 1.0
        return out

    @staticmethod
    def from_tag_name(name: str):
        if name == "non_event":
            return EventType.NON_EVENT
        if name == "change_of_state":
            return EventType.CHANGE_OF_STATE
        if name == "process":
            return EventType.PROCESS
        if name == "stative_event":
            return EventType.STATIVE_EVENT
        raise ValueError(f"Invalid Event variant {name}")

    def to_string(self) -> str:
        if self == EventType.NON_EVENT:
            return "non_event"
        if self == EventType.CHANGE_OF_STATE:
            return "change_of_state"
        if self == EventType.PROCESS:
            return "process"
        if self == EventType.STATIVE_EVENT:
            return "stative_event"
        else:
            ValueError("Unknown EventType")


class SpanAnnotation(NamedTuple):
    text: str
    event_type: EventType
    document_text: str
    # These annotate the start and end offsets in the document string
    start: int
    end: int

    @staticmethod
    def to_batch(data: List[Self], tokenizer: PreTrainedTokenizer):
        encoded = tokenizer.batch_encode_plus(
            [anno.text for anno in data],
            return_tensors="pt",
            padding=True,
        )
        labels = torch.tensor([anno.event_type.value for anno in data])
        return encoded, labels, data

    def output_dict(self, predicted_label):
        return {
            "start": self.start,
            "end": self.end,
            "predicted": EventType(predicted_label).to_string(),
        }


class SimpleEventDataset(Dataset):
    """
    Dataset of all event spans with their features.
    """
    def __init__(self, project: catma.CatmaProject, annotation_collections: Iterable[str] = ()):
        """
        Args:
            project: CatmaProject to load from
            annotation_collections: Iterable of annotation collection names to be included
        """
        super().__init__()
        self.annotations: List[SpanAnnotation] = []
        self.tagset = project.tagset_dict["EvENT-Tagset_3"]
        for collection in [project.ac_dict[coll] for coll in annotation_collections]:
            for annotation in collection.annotations:
                try:
                    span_anno = SpanAnnotation(
                        text=annotation.text,
                        event_type=EventType.from_tag_name(annotation.tag.name),
                        document_text=collection.text,
                        start=annotation.start_point,
                        end=annotation.end_point,
                    )
                    self.annotations.append(span_anno)
                except ValueError as e:
                    print(f"Error parsing span annotation: {e}")

    def __getitem__(self, i: int):
        return self.annotations[i]

    def __len__(self):
        return len(self.annotations)
