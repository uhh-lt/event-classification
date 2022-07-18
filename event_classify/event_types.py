from enum import Enum
import uuid
import torch

class EventType(Enum):
    NON_EVENT = 0
    CHANGE_OF_STATE = 1
    PROCESS = 2
    STATIVE_EVENT = 3

    def to_onehot(self):
        out = torch.zeros(4)
        out[self.value] = 1.0
        return out

    def get_narrativity_ordinal(self):
        if self == EventType.NON_EVENT:
            return 0
        elif self == EventType.STATIVE_EVENT:
            return 1
        elif self == EventType.PROCESS:
            return 2
        elif self == EventType.CHANGE_OF_STATE:
            return 3

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

    def get_tag_uuid(self) -> uuid.UUID:
        if self == EventType.NON_EVENT:
            return uuid.UUID("739E8B44-494A-4B12-9E22-1731B6AB354A")
        if self == EventType.CHANGE_OF_STATE:
            return uuid.UUID("710B4F28-E827-476D-8E32-B8F7287307E3")
        if self == EventType.PROCESS:
            return uuid.UUID("9F68020E-29BA-48D7-916D-C19DD4D446A5")
        if self == EventType.STATIVE_EVENT:
            return uuid.UUID("E4BB23A3-3975-47D9-9157-A0F53DBDCFFF")
        else:
            raise ValueError("Unknown EventType")

