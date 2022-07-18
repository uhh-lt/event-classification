import uuid
from catma_py.catma  import Tag, Tagset

from event_classify.event_types import EventType



def build_tagsets():
    author_prop_uuid = uuid.UUID('29ad924a-a7d2-419a-89fb-fa4d2d0060c2')
    author_name = "Automated Event Annotator"
    tags = {e.value: Tag(e.to_string(), tag_uuid=e.get_tag_uuid()) for e in [EventType.NON_EVENT, EventType.STATIVE_EVENT]}
    parent_tag = Tag("Active Event", tag_uuid=uuid.UUID("D28B4098-B209-4A00-872A-7543DBABCECB"))
    tags.update({
        None: parent_tag,
        EventType.CHANGE_OF_STATE.value: Tag(name=EventType.CHANGE_OF_STATE.to_string(), tag_uuid=EventType.CHANGE_OF_STATE.get_tag_uuid(), parent=parent_tag),
        EventType.PROCESS.value: Tag(name=EventType.PROCESS.to_string(), tag_uuid=EventType.PROCESS.get_tag_uuid(), parent=parent_tag),
    })
    tag_set = Tagset(f"EvENT-Tagset_3", tags=tags.values(), tagset_uuid=uuid.UUID("BA479AEC-D3C3-4CAA-95E0-83ECD239B164"))
    return tags, tag_set