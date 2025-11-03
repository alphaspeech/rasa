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
from rasa.llm_nlu.utils.send_request import fill_prompt_template, get_simplified_history
from rasa.shared.core.events import BotUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

class NaturalLanguageSummarizer(NaturalLanguageGenerator):
    def __init__(self, endpoint_config: EndpointConfig) -> None:

        self.nlg_endpoint = endpoint_config
        self.summarize = True

    def _clean_utter_action(self, utter_action: BotUttered):
        clean_dict = {
            "name": utter_action.metadata.get("utter_action", None),
            "text": utter_action.text,
            "rephrase": utter_action.metadata.get("metadata", {}).get("rephrase", False),
            "rephrase_prompt": utter_action.metadata.get("metadata", {}).get("rephrase_prompt", None),
        }

        return clean_dict

    async def generate(
        self,
        utter_actions: List[BotUttered],
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Dict[Text, Any]:
        """Retrieve a named response from the domain using an endpoint."""
        domain_responses = kwargs.pop("domain_responses", None)
        conversation_history = get_simplified_history(tracker, 5)
        utter_actions = [self._clean_utter_action(a) for a in utter_actions]
        goal = tracker.active_goal_name

        prompt = fill_prompt_template(
            SUMMARIZE_PROMPT,
            goal=goal,
            user_input=tracker.latest_message.text,
            conversation_history=conversation_history,
            responses=utter_actions,
        )

        logger.info(prompt)

        model = self.nlg_endpoint.kwargs.get("model", "qwen2.5:14b")

        request_body = {"model": model, "stream": False, "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}]}

        logger.debug(
            "Requesting NLG for {} from {}."
            "The request prompt is {}."
            "".format(utter_actions, self.nlg_endpoint.url, prompt)
        )

        response = await self.nlg_endpoint.request(
            method="post", json=request_body, timeout=DEFAULT_REQUEST_TIMEOUT
        )

        logger.info(response)

        logger.debug(f"Received NLG response: {response}")

        return response["message"]["content"]



SUMMARIZE_PROMPT = """
    Current goal name: {goal}
    
    Consider the conversation history to generate a suited reply. "user" is the user, "bot" is your replies. The conversation history is:
    {conversation_history}
    
    The latest user input was:
    {user_input}
    
    Generate a reply to return to the user based on these actions you are supposed to perform next:
    {responses}
    
    Use the text parameter provided with the utter actions as a basis for what you're supposed to say. 
    - If "rephrase" is True, you're allowed to rephrase the text to make it sound more natural. If a rephrase prompt is available, consider them like instructions.
    - If rephrase is False, output the text exactly as defined for the utter action.
    - If there are multiple utter actions, ensure that you are generating a cohesive reply including all of them. Avoid rephrasing for utter actions with "rephrase" False.
    Only add a greeting if this is the first bot message in the conversation history.
    
    Now generate your reply.
"""