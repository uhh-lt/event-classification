from dataclasses import dataclass
from typing import List
from omegaconf import MISSING
from enum import Enum


class Output:
    THOUGHT_REPRESENTATION = "thought_representation"
    MENTAL = "mental"
    ITERATIVE = "iterative"
    SPEECH = "speech"
    EVENT_KIND = "event_kind"


@dataclass
class DatasetConfig():
    """
    Config details relevant to the dataset to use
    """
    catma_uuid: str
    catma_dir: str
    in_distribution: bool
    special_tokens: bool
    excluded_collections: List[str]


@dataclass
class SchedulerConfig():
    """
    Configuration for the learning rate scheduler.
    """
    enable: bool
    # The number of epochs to scale the learning rate scheduler over
    epochs: int

@dataclass
class Config():
    """
    Config for the entire program, holding hyperparameters etc.
    """
    device: str
    optimize: str
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    scheduler: SchedulerConfig
    dataset: DatasetConfig
    pretrained_model: str
    label_smoothing: bool
    loss_report_frequency: int
    optimize_outputs: List[Output]
    # If true static_loss_weights are disregarded
    dynamic_loss_weighting: bool
    static_loss_weights: List[float]
