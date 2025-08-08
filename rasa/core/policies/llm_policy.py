from __future__ import annotations
import copy
import logging

from typing import Any, Dict, Optional, Text, Tuple, Union, List, Type

from pydantic import BaseModel

from rasa.core.constants import LLM_POLICY_PRIORITY, POLICY_PRIORITY
from rasa.core.policies.multi_policy import MultiPolicy
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.core.policies.policy import PolicyPrediction, Policy, SupportedData
from rasa.graph_components.providers.domain_for_core_training_provider import DomainForCoreTrainingProvider
from rasa.graph_components.providers.goals_provider import GoalsProvider
from rasa.graph_components.providers.responses_provider import ResponsesProvider
from rasa.llm_nlu.utils.send_request import fill_prompt_template, send_request, get_simplified_history
from rasa.shared.constants import DIAGNOSTIC_DATA
import rasa.shared.utils.io
from rasa.shared.core.constants import ACTIVE_GOAL, DEFAULT_ACTION_NAMES, ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered, BotUttered
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data import util
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.classifiers.diet_classifier import (
    DIETClassifier,
)
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.nlu.constants import (
    RESPONSE_SELECTOR_PROPERTY_NAME,
    RESPONSE_SELECTOR_RETRIEVAL_INTENTS,
    RESPONSE_SELECTOR_RESPONSES_KEY,
    RESPONSE_SELECTOR_PREDICTION_KEY,
    RESPONSE_SELECTOR_RANKING_KEY,
    RESPONSE_SELECTOR_UTTER_ACTION_KEY,
    RESPONSE_SELECTOR_DEFAULT_INTENT,
    DEFAULT_TRANSFORMER_SIZE,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    RESPONSE,
    INTENT_RESPONSE_KEY,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)

from rasa.utils.tensorflow.models import RasaModel

logger = logging.getLogger(__name__)

REMEMBER_LATEST_NUM_KEY = "remember_latest_num"
USE_STORIES_KEY = "use_stories"
USE_GOALS_KEY = "use_goals"
MODEL_KEY = "model"

REMEMBER_LATEST_NUM_DEFAULT = 5
USE_STORIES_DEFAULT = False
USE_GOALS_DEFAULT = True
MODEL_DEFAULT = "qwen2.5:14b"

ENDPOINT = "endpoint"
MAX_CHAR = "max_char"
STREAM = "stream"
TEXT = "text"

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=False
)
class LLMPolicy(MultiPolicy, GoalsProvider, ResponsesProvider):
    """Response selector using llm to predict next action
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return []

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            POLICY_PRIORITY: LLM_POLICY_PRIORITY,
            REMEMBER_LATEST_NUM_KEY: REMEMBER_LATEST_NUM_DEFAULT, # use the latest REMEMBER_LATEST_NUM messages for context
            # ENDPOINT: "https://llm.dev.alphaspeech.de/api/chat", # TODO: Get from endpoints.yml
            # MODEL_KEY: MODEL_DEFAULT,
            USE_STORIES_KEY: USE_STORIES_DEFAULT,
            USE_GOALS_KEY: USE_GOALS_DEFAULT,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs
    ) -> None:
        logging.info("################## INITIALIZED LLMPolicy ###")
        logging.info(kwargs)
        logging.info(execution_context)
        # TODO: Consider rules and stories in the future
        # TODO: Consider being outside of a goal too

        # Initialize defaults
        # self.responses = responses or {}

        self.responses = None
        self.goals = None

        super().__init__(
            config,
            model_storage,
            resource,
            execution_context,
        )

        self.priority = config.get(POLICY_PRIORITY, LLM_POLICY_PRIORITY)


    # def __determine_current_goal(self, user_input):
    #     """
    #     Determine the current goal of the conversation and set current goal.
    #
    #     :param user_input:
    #     :return:
    #     """
    #     formatted_goals = {}
    #     for g_key, g_value in self.goals.items():
    #         formatted_goals[g_key] = {
    #             "description": g_value.get("description")
    #         }
    #
    #     current_goal = None
    #     if self.current_goal is not None:
    #         current_goal = self.current_goal + ": " + self.goals[self.current_goal]["description"]
    #
    #     template = fill_prompt_template(
    #         DETERMINE_GOAL_TEMPLATE,
    #         goal=current_goal,
    #         user_input=user_input,
    #         goals=formatted_goals
    #     )
    #
    #     new_goal_id = send_request(template, GoalExtraction.model_json_schema())["goal_key"]
    #     print(f"Predicted current goal: {new_goal_id}")
    #
    #     self.current_goal = new_goal_id

    def _predict(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> List[Text]:
        current_tracker_state = tracker.current_state()
        logging.info(current_tracker_state)

        user_input = current_tracker_state["latest_message"]["text"]
        conversation_history = get_simplified_history(tracker, self.config[REMEMBER_LATEST_NUM_KEY])

        active_goal = current_tracker_state[ACTIVE_GOAL]
        if active_goal == {}:
            active_goal = None

        if current_tracker_state["latest_action_name"] == ACTION_LISTEN_NAME:
            logging.info("should update goal") # TODO: update goal here

        prompt = fill_prompt_template(
            PREDICT_ACTIONS_PROMPT_TEMPLATE,
            goal=active_goal,
            steps=active_goal,
            user_input=user_input,
            available_actions=self._get_available_actions(domain),
            conversation_history=conversation_history,
            default_actions=DEFAULT_ACTION_NAMES,
            action_listen_name=ACTION_LISTEN_NAME,
        )
        logging.info(prompt)
        data = send_request(prompt, ActionPrediction.model_json_schema())

        next_actions = [action_key for action_key in data.get("action_keys")]
        return next_actions

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: The tracker containing the conversation history up to now.
            domain: The model's domain.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to make predictions.

        Returns:
             The prediction.
        """
        # Check first if there is already next predictions stored
        try:
            stored_next_prediction = tracker.recall_next_prediction()
            if stored_next_prediction is not None:
                result = self._prediction_result(stored_next_prediction, tracker, domain)
                return self._prediction(result)
        except TypeError:
            logger.warning(
                f"LLMPolicy ran out of stored predictions for the" 
                "latest user query. Returning action_listen."
            )
            listen_prediction = self._prediction_result(ACTION_LISTEN_NAME, tracker, domain)
            return self._prediction(listen_prediction)

        # make fresh prediction if no predictions stored
        result = self._default_predictions(domain)

        predicted_action_names = self._predict(
            tracker,
            domain,
            rule_only_data,
            **kwargs
        )
        logging.info(f"PREDICTED ACTIONS: {predicted_action_names}")

        if predicted_action_names is not None:
            tracker.remember_predictions(predicted_action_names)

            result = self._prediction_result(tracker.recall_next_prediction(), tracker, domain)
        else:
            logger.debug("LLMPolicy could not determine next action.")

        return self._prediction(result)

    def _prediction_result(
        self, action_name: Text, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        result = self._default_predictions(domain)
        result[domain.index_for_action(action_name)] = 1.0

        return result

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:
        """Trains a policy.

        Args:
            training_trackers: The story and rules trackers from the training data.
            domain: The model's domain.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to train itself.

        Returns:
            A policy must return its resource locator so that potential children nodes
            can load the policy from the resource.
        """
        logging.info("KWARGS HERE:")
        logging.info(kwargs)
        logging.info(domain)
        logging.info(training_trackers)
        logging.info(self._resource)

        self.responses = domain.responses

        return self._resource


    def _get_available_actions(self, domain: Domain) -> List[Text]:
        action_names = domain.action_names_or_texts
        action_names = [action_name for action_name in action_names if action_name not in DEFAULT_ACTION_NAMES]
        logging.info("######## available actions: ##########")
        logging.info(action_names)
        return action_names



PREDICT_ACTIONS_PROMPT_TEMPLATE = """
    {goal}

    For context, here is the conversation history. 
    - "user" is the user
    - "bot" is your replies.
    {conversation_history}
    The latest user input was:
    {user_input}

    Using the conversation history, determine which step to take next. Here are the steps to take to achieve the current goal.
    Skip steps that are already done or unnecessary considering the conversation history.
    Only do one step at a time.
    {steps}

    Once you know which step to take, output a list of actions to perform in order to complete it.
    Each action shall be referred to by its unique name: the action key.
    If the step says to collect certain parameter, the matching action key would be utter_ask_<param>. e.g. to collect "name", use the key 'utter_ask_name' to ask the user for their name.
    But again, skip collect steps if the information to collect is already available.

    Here are the available actions:
    {available_actions}
    
    The available actions are use case specific. Here are actions that are always available and should be picked with lower priority:
    {default_actions}

    For example, if you determined that the step "collect: appointment_type" is the next step, you would make a list of actions to do to collect the appointment type.
    That would be:
    - utter_ask_appointment_type
    - {action_listen_name}
    You would ignore steps that were already completed and steps that come after.

    Return ONLY the list of action keys needed to complete the step you determined. The last action should always be to wait for the next user input.
"""

DETERMINE_GOAL_TEMPLATE = """
    Please help determine whether the current goal of the conversation has changed considering the latest user input.

    The current goal is {goal}.

    The available goals are:
    {goals}

    The latest user input is:
    {user_input}

    Now output the current goal of the conversation. If the goal has not changed, output the current goal.
"""

class GoalExtraction(BaseModel):
    goal_key: str

class ActionPrediction(BaseModel):
    action_keys: List[str]
