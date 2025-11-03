import json
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, create_model
import requests

from rasa.shared.core.events import ActionExecuted, BotUttered, UserUttered
#from rasa.shared.core.trackers import DialogueStateTracker # This import causes error during training

# TODO: load url, model, temperature from config
url = "https://llm.dev.alphaspeech.de/api/chat"

def send_request(prompt: str, format: dict[str, Any] = None, model: str = "qwen2.5:14b"):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": model,  # "cyberwald/sauerkrautlm-nemo-12b-instruct",
        "stream": False,
        "temperature": 0
    }

    if format is not None:
        data["format"] = format

    data["messages"] = [{"role": "user", "content": prompt}]

    response = requests.post(url, headers=headers, data=json.dumps(data))

    content = json.loads(response.content.decode('utf-8'))["message"]["content"]

    if format is not None:
        return json.loads(content)

    else:
        return content

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
            conversation_history.append({
                "role": "bot",
                "action": event.text
            })
        if type(event) == UserUttered:
            conversation_history.append({
                "role": "user",
                "message": event.text
            })

    return list(reversed(conversation_history))


def create_slot_extraction_model(slots):
    fields = {}

    def get_type(s_type):
        if s_type == "text":
            return Optional[str], None
        return Optional[Any], None

    for slot_object in slots:
        slot_mappings = slot_object.mappings

        from_entity_mappings = [m for m in slot_mappings if m.get("type") == "from_entity"]
        for from_entity_m in from_entity_mappings:
            slot_type = slot_object.type_name
            entity_name = from_entity_m.get("entity")
            fields[entity_name] = get_type(slot_type) if slot_type else Optional[str] # TODO: ensure fields works with rasa types

    return create_model("LLMSlotExtraction", **fields, __base__=BaseModel)