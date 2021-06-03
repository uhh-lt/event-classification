from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class DatasetConfig():
    """
    Config details relevant to the dataset to use
    """
    catma_uuid: str
    catma_dir: str
    in_domain: bool


@dataclass
class Config():
    """
    Config for the entire program, holding hyperparameters etc.
    """
    device: str
    optimize: str
    epochs: int
    # The number of epochs to scale the learning rate scheduler over
    lr_scheduler_epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    dataset: DatasetConfig
    pretrained_model: str
