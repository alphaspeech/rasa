import json
import logging
from enum import Enum
from typing import List, Any, Dict, Optional, Type, Generic, TypeVar
from pydantic import BaseModel, create_model
import requests

logger = logging.getLogger(__name__)
T = TypeVar("T")

from rasa.shared.core.events import ActionExecuted, BotUttered, UserUttered
#from rasa.shared.core.trackers import DialogueStateTracker # This import causes error during training

# TODO: load url, model, temperature from config
url = "https://llm.dev.alphaspeech.de/api/generate"

def send_request(prompt: str, format: dict[str, Any] = None, model: str = "qwen2.5:14b"):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": model,  # "cyberwald/sauerkrautlm-nemo-12b-instruct",
        "stream": False,
        "prompt": prompt,
        # "think": True
    }

    if format is not None:
        data["format"] = format
    logger.info(data)
    response = requests.post(url, headers=headers, data=json.dumps(data))

    content = json.loads(response.content.decode('utf-8'))["response"]
    logger.info(content)
    if format is not None:
        return json.loads(content)

    else:
        return content

def convert_history_to_llm_messages(tracker, remember_latest: int)-> List[Dict]:
    messages = []

    events = tracker.events
    if not remember_latest or len(events) < remember_latest:
        remember_latest = len(events)

    for event in reversed(events):
        if len(messages) >= remember_latest:
            break

        if type(event) == BotUttered and len(messages) > 0: # do not include latest bot messages
            messages.append({
                "role": "assistant",
                "content": event.text
            })
        if type(event) == UserUttered:
            messages.append({
                "role": "user",
                "content": event.text
            })

    return list(reversed(messages))


def fill_prompt_template(template: str, **params) -> str:
    return template.format(**params)

def get_simplified_history(tracker, remember_latest: int) -> List[Dict]:
    """
    Returns a simplified conversation history to use for prompting.
    tracker = DialogueStateTracker
    """
    conversation_history = []

    events = tracker.events
    if len(events) < remember_latest:
        remember_latest = len(events)

    for event in reversed(events):
        if len(conversation_history) >= remember_latest:
            break

        if type(event) == BotUttered:
            conversation_history.append(f"Bot: {event.text}\n")
        if type(event) == UserUttered:
            conversation_history.append(f"User: {event.text}\n")

    return list(reversed(conversation_history))


class Judgement(BaseModel, Generic[T]):
    value: Optional[T]
    certainty: float

def create_slot_extraction_model(slots):
    fields = {}

    def as_judgement_type(inner_type: Type):
        return Optional[Judgement[inner_type]]

    def get_type(s_type, s_object):
        if s_type == "text":
            return str
        if s_type == "bool":
            return bool
        if s_type == "float":
            return float
        if s_type == "categorical":
            cat_values = s_object.values
            enum_cls = Enum(
                s_object.type_name.capitalize(),
                {v: v for v in cat_values},
                type=str
            )
            return enum_cls
        return Any

    for slot_object in slots:
        slot_mappings = slot_object.mappings

        from_entity_mappings = [m for m in slot_mappings if m.get("type") in "from_entity"]
        for from_entity_m in from_entity_mappings:
            slot_type = slot_object.type_name
            entity_name = from_entity_m.get("entity") or from_entity_m.name
            base_type = get_type(slot_type, slot_object)
            if slot_object.judgement_certainty is not None:
                field_type = as_judgement_type(base_type)
            else:
                field_type = Optional[base_type]
            fields[entity_name] = (field_type, None)

    return create_model("LLMSlotExtraction", **fields, __base__=BaseModel)