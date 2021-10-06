from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import copy
from enum import Enum
import json
import logging
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple, Any

import catma_gitlab as catma
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class EventClassificationLabels:
    event_kind: torch.Tensor
    iterative: torch.Tensor
    speech_type: torch.Tensor
    thought_representation: torch.Tensor
    mental: torch.Tensor

    def to(self, device):
        new = copy.deepcopy(self)
        for member in new.__dataclass_fields__:
            setattr(new, member, getattr(new, member).to(device))
        return new


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

    def get_narrativity_score(self):
        if self == EventType.NON_EVENT:
            return 0
        if self == EventType.CHANGE_OF_STATE:
            return 7
        if self == EventType.PROCESS:
            return 5
        if self == EventType.STATIVE_EVENT:
            return 2
        else:
            raise ValueError("Unknown EventType")

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
            raise ValueError("Unknown EventType")


class SpeechType(Enum):
    CHARACTER = 0
    NARRATOR = 1
    NONE = 2

    def to_onehot(self, device="cpu"):
        out = torch.zeros(3, device=device)
        out[self.value] = 1.0
        return out

    @staticmethod
    def from_list(in_list: List[str]) -> SpeechType:
        if "character_speech" in in_list:
            return SpeechType.CHARACTER
        elif "narrator_speech" in in_list:
            return SpeechType.CHARACTER
        elif len(in_list) > 0:
            return SpeechType.NONE
        else:
            raise ValueError("RepresentationType not specified")


class RepresentationType(Enum):
    THOUGHT = 0
    CHARACTER_SPEECH = 1
    NARRATOR_SPEECH = 2
    THOUGHT_CHARACTER_SPEECH = 3
    THOUGHT_NARRATOR_SPEECH = 4

    def to_onehot(self):
        out = torch.zeros(4)
        out[self.value] = 1.0
        return out

    @staticmethod
    def from_list(representation_list):
        if len(representation_list) > 2:
            raise ValueError("Representation list may only have a maximum of two values")
        if "thought_representation" in representation_list:
            if len(representation_list) == 2:
                if "narrator_speech" in representation_list:
                    return RepresentationType.THOUGHT_CHARACTER_SPEECH
                elif "character_speech" in representation_list:
                    return RepresentationType.THOUGHT_CHARACTER_SPEECH
                else:
                    raise ValueError("Invalid combination of representations")
            else:
                return RepresentationType.THOUGHT
        else:
            if len(representation_list) != 1:
                raise ValueError("Only `thought_representation` allows for multiple values in representation list.")
            elif representation_list == ["narrator_speech"]:
                return RepresentationType.NARRATOR_SPEECH
            elif representation_list == ["character_speech"]:
                return RepresentationType.CHARACTER_SPEECH
            else:
                raise ValueError("Invalid representation type")




class SpanAnnotation(NamedTuple):
    text: str
    special_token_text: str
    iterative: Optional[bool]
    speech_type: SpeechType
    thought_representation: bool
    mental: Optional[bool]
    event_type: Optional[EventType]
    document_text: str
    # These annotate the start and end offsets in the document string
    start: int
    end: int
    spans: List[Tuple[int, int]]

    @staticmethod
    def to_batch(data: List[SpanAnnotation], tokenizer: PreTrainedTokenizer):
        encoded = tokenizer.batch_encode_plus(
            [anno.special_token_text for anno in data],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if any(anno.event_type is None for anno in data):
            labels = None
        else:
            labels = EventClassificationLabels(
                event_kind=torch.tensor([anno.event_type.value for anno in data]),
                mental=torch.tensor([anno.mental for anno in data if anno.event_type != EventType.NON_EVENT], dtype=torch.float),
                iterative=torch.tensor([anno.iterative for anno in data if anno.event_type != EventType.NON_EVENT], dtype=torch.float),
                speech_type=torch.tensor([anno.speech_type.value for anno in data]),
                thought_representation=torch.tensor([anno.thought_representation for anno in data], dtype=torch.float),
            )
        return encoded, labels, data

    @staticmethod
    def build_special_token_text(annotation: catma.Annotation, document, include_special_tokens: bool = True):
        output = []
        plain_text = document.plain_text
        # Provide prefix context
        previous_end = annotation.start_point - 100
        for selection in merge_direct_neighbors(copy.deepcopy(annotation.selectors)):
            output.append(plain_text[previous_end:selection.start])
            if include_special_tokens:
                output.append(" <SE> ")
            output.append(plain_text[selection.start:selection.end])
            if include_special_tokens:
                output.append(" <EE> ")
            previous_end = selection.end
        # Provide suffix context
        output.append(plain_text[previous_end:previous_end + 100])
        return "".join(output)

    @staticmethod
    def build_special_token_text_from_json(annotation: Dict, document: str, include_special_tokens: bool = True):
        selections = [tuple(span) for span in annotation["spans"]]
        output = []
        # Provide prefix context
        previous_end = annotation["start"] - 100
        for start, end in selections:
            output.append(document[previous_end:start])
            if include_special_tokens:
                output.append(" <SE> ")
            output.append(document[start:end])
            if include_special_tokens:
                output.append(" <EE> ")
            previous_end = end
        # Provide suffix context
        output.append(document[previous_end:previous_end + 100])
        return "".join(output)

    def output_dict(self, predicted_label):
        return {
            "start": self.start,
            "end": self.end,
            "spans": self.spans,
            "predicted": EventType(predicted_label).to_string(),
        }


def simplify_representation(repr_list):
    repr_list = [t.replace("_1", "").replace("_2", "_").replace("_3", "") for t in repr_list]
    return repr_list


class SimpleEventDataset(Dataset):
    """
    Dataset of all event spans with their features.
    """
    def __init__(self, project: catma.CatmaProject, annotation_collections: Iterable[str] = (), include_special_tokens: bool = True):
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
                if annotation.tag.name in ["Zweifelsfall", "change_of_episode"]:
                    continue # We ignore these
                if len(annotation.properties.get("mental", [])) > 1:
                    logging.warning("Ignoring annotation with inconsistent 'mental' property")
                    continue
                try:
                    special_token_text = SpanAnnotation.build_special_token_text(
                        annotation,
                        collection.text,
                        include_special_tokens=include_special_tokens,
                    )
                    simple_representations = simplify_representation(annotation.properties["representation_type"])
                    speech_type = SpeechType.from_list(simple_representations)
                    thought_representation = "thought_representation" in simple_representations
                    if "narrator_speech" in annotation.properties["representation_type"]:
                        pass
                    event_type = EventType.from_tag_name(annotation.tag.name)
                    iterative = annotation.properties.get("iterative", ["no"]) == ["yes"]
                    mental = annotation.properties.get("mental", ["no"]) == ["yes"]
                    if event_type == EventType.NON_EVENT:
                        iterative = None
                        mental = None
                    span_anno = SpanAnnotation(
                        text=annotation.text,
                        special_token_text=special_token_text,
                        event_type=event_type,
                        iterative=iterative,
                        speech_type=speech_type,
                        thought_representation=thought_representation,
                        mental=mental,
                        document_text=collection.text.plain_text,
                        start=annotation.start_point,
                        end=annotation.end_point,
                        spans=[(s.start, s.end) for s in merge_direct_neighbors(copy.deepcopy(annotation.selectors))],
                    )
                    self.annotations.append(span_anno)
                except ValueError as e:
                    logging.warning(f"Error parsing span annotation: {e}")

    def __getitem__(self, i: int):
        return self.annotations[i]

    def __len__(self):
        return len(self.annotations)


class JSONDataset(Dataset):
    """
    Dataset based on JSON file created by our preprocessing script
    """
    def __init__(self, dataset_file: Optional[str], data : Optional[list] = None, include_special_tokens: bool = True):
        """
        Args:
            dataset_file: Path to json file created by preprocessing script
            data: Instead of a file path read data from this dict instead
        """
        super().__init__()
        self.annotations : List[SpanAnnotation] = []
        self.documents: defaultdict[str, List[SpanAnnotation]] = defaultdict(list)
        if data is None:
            if dataset_file is None:
                raise ValueError("Only one of dataset_file and data may be None")
            else:
                data = json.load(open(dataset_file))
        for document in data:
            title = document["title"]
            full_text = document["text"]
            for annotation in document["annotations"]:
                special_token_text = SpanAnnotation.build_special_token_text_from_json(
                    annotation, full_text, include_special_tokens
                )
                event_type = None
                if annotation.get("prediction") is not None:
                    event_type = EventType.from_tag_name(annotation["predicted"])
                text = full_text[annotation["start"]:annotation["end"]]
                span_anno = SpanAnnotation(
                    text=text,
                    special_token_text=special_token_text,
                    event_type=event_type,
                    document_text=full_text,
                    start=annotation["start"],
                    end=annotation["end"],
                    spans=[(s[0], s[1]) for s in annotation["spans"]],
                )
                self.documents[title].append(span_anno)
                self.annotations.append(span_anno)

    def get_annotation_json(self, predictions: List[EventType]) -> List[Dict[str, Any]]:
        out_data = []
        i = 0
        if len(predictions) > 0 and len(predictions) != len(self.annotations):
            raise ValueError("Prediction list should be the length of the list of annotations")
        for title, document in self.documents.items():
            out_doc = {
                "title": title,
                "text": None,
                "annotations": []
            }
            for annotation in document:
                if out_doc["text"] is None:
                    out_doc["text"] = annotation.document_text
                try:
                    prediction = predictions[i]
                except IndexError:
                    prediction = None
                out_doc["annotations"].append(annotation.output_dict(prediction))
                i += 1
            out_doc["annotations"] = list(sorted(out_doc["annotations"], key=lambda d: d["start"]))
            out_data.append(out_doc)
        return out_data

    def save_json(self, out_path: str, predictions: List[EventType] = []):
        out_data = self.get_annotation_json(predictions)
        out_file = open(out_path, "w")
        json.dump(out_data, out_file)

    def __getitem__(self, i: int):
        return self.annotations[i]

    def __len__(self):
        return len(self.annotations)


def merge_direct_neighbors(selectors):
    to_remove = []
    for i in range(len(selectors) - 1):
        if selectors[i].end == selectors[i + 1].start:
            selectors[i + 1].start = selectors[i].start
            to_remove.append(i)
    return [selector for i, selector in enumerate(selectors) if i not in to_remove]
