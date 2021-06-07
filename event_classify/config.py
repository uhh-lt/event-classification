from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class DatasetConfig():
    """
    Config details relevant to the dataset to use
    """
    catma_uuid: str
    catma_dir: str
    in_distribution: bool


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
