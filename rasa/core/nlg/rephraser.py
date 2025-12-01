import json
import logging
from typing import (
    List,
    Text,
    Dict,
    Any, Optional,
)

from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.nlg.constants import REPHRASE_PROMPTS, EN_KEY, DE_KEY
from rasa.llm_nlu.utils.send_request import fill_prompt_template, get_simplified_history, \
    convert_history_to_llm_messages
from rasa.shared.core.domain import Domain, KEY_PROMPT
from rasa.shared.core.events import BotUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

REPHRASER_KEY = "rephraser"
REPHRASE_KEY = "rephrase"
REPHRASE_PROMPT_KEY = "rephrase_prompt"
DESCRIPTION_KEY = "description"

URL_KEY = "url"
ENABLED_KEY = "enabled"
LANGUAGE_KEY = "language"
MODEL_KEY = "model"
DEFAULT_MODEL = "qwen2.5:14b"

class NaturalLanguageRephraser(NaturalLanguageGenerator):
    def __init__(self, endpoint_config: EndpointConfig, domain: Optional[Domain] = None) -> None:

        self.nlg_endpoint = endpoint_config
        self.url = endpoint_config.url
        self.enabled = endpoint_config.params.get(ENABLED_KEY, True)
        self.language = endpoint_config.params.get(LANGUAGE_KEY, EN_KEY)
        self.model = endpoint_config.params.get(MODEL_KEY, DEFAULT_MODEL)
        self.system_prompt = domain.prompt


    def _clean_utter_action(self, utter_action: BotUttered, tracker: DialogueStateTracker):
        clean_dict = {
            "name": utter_action.metadata.get("utter_action", None),
            "text": utter_action.text,
            DESCRIPTION_KEY: utter_action.metadata.get("metadata", {}).get(DESCRIPTION_KEY, None),
            REPHRASE_KEY: utter_action.metadata.get("metadata", {}).get(REPHRASE_KEY, False),
            REPHRASE_PROMPT_KEY: utter_action.metadata.get("metadata", {}).get(REPHRASE_PROMPT_KEY, ""),
        }

        clean_dict[REPHRASE_PROMPT_KEY] = clean_dict[REPHRASE_PROMPT_KEY].format(**tracker.current_slot_values())

        return clean_dict

    async def generate(
        self,
        utter_actions: List[BotUttered],
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Dict[Text, Any]:
        """Retrieve a named response from the domain using an endpoint."""
        conversation_history = convert_history_to_llm_messages(tracker, None)
        utter_actions = [self._clean_utter_action(a, tracker) for a in utter_actions]

        # skip prompting if all utter actions have rephrase=false
        skip_response = ""
        for utter_action in utter_actions:
            if utter_action[REPHRASE_KEY] and self.enabled:
                skip_response = None
                break
            skip_response = skip_response + utter_action["text"] + " "
        if skip_response is not None:
            return skip_response.strip()

        # generate response from one or multiple utter actions
        prompt = fill_prompt_template(
            REPHRASE_PROMPTS.get(self.language, REPHRASE_PROMPTS[DE_KEY]),
            system_prompt=self.system_prompt,
            user_input=tracker.latest_message.text,
            responses=utter_actions,
        )

        conversation_history = [{"role": "system", "content": prompt}] + conversation_history
        request_body = {
            "model": self.model,
            "stream": False,
            "temperature": 0,
            "messages": conversation_history
        }

        logger.debug(
            "Requesting NLG for {} from {}."
            "The request prompt is {}."
            "".format(utter_actions, self.nlg_endpoint.url, prompt)
        )

        response = await self.nlg_endpoint.request(
            method="post", json=request_body, timeout=DEFAULT_REQUEST_TIMEOUT
        )

        logger.debug(f"Received NLG response: {response}")

        return response["message"]["content"]



