from transformers import ElectraTokenizer, ElectraPreTrainedModel, ElectraModel
from typing import Optional
import torch
from torch import nn
from dataclasses import dataclass
from event_classify.label_smoothing import LabelSmoothingLoss
from event_classify.datasets import EventClassificationLabels


EVENT_CLASS_WEIGHTS = torch.tensor([0.0003, 0.15, 0.0003, 0.0005])
EVENT_PROPERTIES = {
    "categories": 4,
    "iterative": 1,
    "character_speech": 3,
    "thought_representation": 1,
    "mental": 1,
}


@dataclass
class EventClassificationOutput():
    event_type: torch.Tensor
    iterative: torch.Tensor
    speech_type: torch.Tensor
    thought_representation: torch.Tensor
    mental: torch.Tensor
    loss: Optional[torch.Tensor] = None


class ClassificationHead(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraForEventClassification(ElectraPreTrainedModel):
    def __init__(self, config, label_smoothing: bool = True):
        super().__init__(config)
        if label_smoothing:
            self.event_type_kind_loss = LabelSmoothingLoss(weight=EVENT_CLASS_WEIGHTS)
        else:
            self.event_type_loss = nn.CrossEntropyLoss()
        self.property_loss = nn.CrossEntropyLoss()
        self.electra = ElectraModel(config)
        self.config = config
        self.event_type = ClassificationHead(config, num_labels=EVENT_PROPERTIES["categories"])
        self.iterative = ClassificationHead(config, num_labels=EVENT_PROPERTIES["iterative"])

        self.thought_embedding = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.character_speech = nn.Sequential(
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, EVENT_PROPERTIES["character_speech"]),
        )
        self.thought_representation = nn.Sequential(
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, EVENT_PROPERTIES["thought_representation"]),
        )
        self.mental = nn.Sequential(
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, EVENT_PROPERTIES["mental"]),
        )
        self.binary_criterion = nn.BCEWithLogitsLoss()
        self.speech_criterion = nn.CrossEntropyLoss()
        self.event_type_critereon = nn.CrossEntropyLoss(weight=EVENT_CLASS_WEIGHTS)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels: Optional[EventClassificationLabels]=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        )

        sequence_output = discriminator_hidden_states[0]
        logits_kind = self.event_type(sequence_output)
        logits_iterative = self.iterative(sequence_output)
        thought_embedding = self.thought_embedding(sequence_output[:, 0, :])
        logits_speech = self.character_speech(thought_embedding)
        logits_thought_representation = self.mental(thought_embedding)
        logits_mental = self.mental(thought_embedding)

        loss = None
        if labels is not None:
            loss = self.event_type_critereon(logits_kind, labels.event_type)
            loss += self.binary_criterion(logits_thought_representation.squeeze(), labels.thought_representation)
            loss += self.speech_criterion(logits_speech, labels.speech_type)
            # only for all events that are not non events
            mental_defined = torch.masked_select(logits_mental.squeeze(), labels.event_type != 0)
            loss += self.binary_criterion(mental_defined, labels.mental)
            iterative_defined = torch.masked_select(logits_iterative.squeeze(), labels.event_type != 0)
            loss += self.binary_criterion(iterative_defined, labels.iterative)

        return EventClassificationOutput(
            loss=loss,
            event_type=torch.argmax(logits_kind, 1),
            iterative=torch.argmax(logits_iterative, 1),
            speech_type=torch.argmax(logits_speech, 1),
            thought_representation=torch.argmax(logits_speech, 1),
            mental=torch.argmax(logits_mental, 1),
        )
