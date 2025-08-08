import json
from typing import List, Any, Dict
import requests

from rasa.shared.core.events import ActionExecuted, BotUttered, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker

# TODO: load url, model, temperature from config
url = "https://llm.dev.alphaspeech.de/api/chat"

def send_request(prompt: str, format: dict[str, Any] = None):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": "qwen2.5:14b",  # "cyberwald/sauerkrautlm-nemo-12b-instruct",
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

def get_simplified_history(tracker: DialogueStateTracker, remember_latest: int) -> List[Dict]:
    """
    Returns a simplified conversation history to use for prompting.
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