from __future__ import annotations
import logging
import traceback
from typing import Any, List, Text, Dict, Type, Union, Tuple, Optional

from tensorflow.python.keras.saving.saving_utils import model_metadata

import rasa.shared.utils.io
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    REQUIRE_ENTITIES_KEY, ENTITY_ATTRIBUTE_ROLE, ENTITY_ATTRIBUTE_GROUP,
)
from rasa.core.constants import (
    DEFAULT_UTTERANCE_REJECTION_ENABLED_VALUE,
    DEFAULT_UTTERANCE_REJECTION_THRESHOLD,
    DEFAULT_UTTERANCE_REJECTION_AMBIGUITY_THRESHOLD,
    DEFAULT_UTTERANCE_REJECTION_UPON_UNEXPECTED_THRESHOLD,
    DEFAULT_UTTERANCE_REJECTION_METHOD,
    DEFAULT_UTTERANCE_REJECTION_FORCE_FINAL_VALUE,
    DEFAULT_UTTERANCE_REJECTION_EXCLUDED_INTENTS
)
from rasa.shared.constants import DEFAULT_UTTERANCE_REJECTION_INTENT_NAME, DOMAIN_SCHEMA_FILE

# maybe add intent reject to which the response is always listen? how ensure its not tracked?

ENABLED_KEY = "enabled"
THRESHOLDS_KEY = "threshold"
METHOD_KEY = "method"
FORCE_FINAL_ENABLED_KEY = "force_final_enabled"
EXCLUDED_INTENTS_KEY = "excluded_intents"
REQUIRE_ENTITIES_ENABLED_KEY = "require_entities_enabled"
EFAULT_UTTERANCE_REJECTION_REQUIRE_ENTITIES_ENABLED_VALUE = False

DEFAULT_THRESHOLD_KEY = "default"
AMBIGUITY_THRESHOLD_KEY = "ambiguity_threshold"
UNEXPECTED_UPON_REQUESTED_SLOT_KEY = "unexpected_upon_requested_slot"

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False
)

class UtteranceRejectionClassifier(GraphComponent, IntentClassifier):
    """Handles incoming messages with streaming mode on."""

    def __init__(self, config: Dict[Text, Any], domain: Optional[Domain] = None):
        """Constructs a new utterance rejection classifier."""
        self.component_config = config
        self._domain = domain
        print("Init Rejection Classifier")
        logging.info(domain)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> UtteranceRejectionClassifier:
        """Creates a new component (see parent class for full docstring)."""
        logging.info("CREATING URC")
        return cls(config)

    def process(self, messages: List[Message]) -> List[Message]:
        """Process a list of incoming messages.

        This is the component's chance to process incoming
        messages. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.process`
        of components previous to this one.

        Args:
            messages: List containing :class:
            `rasa.shared.nlu.training_data.message.Message` to process.
        """
        # stack = traceback.format_stack()
        # logging.info("Call stack leading to this function call:")
        # for line in stack[:-1]:  # Exclude the last line because it's the current function call
        #     logging.info(line.strip())

        logging.info(f"PROCESSING MSG IN UTTERANCE REJECTION CLASSIFIER")
        for message in messages:
            if self._full_utterance_detected(message):
                logging.info(f"MESSAGE DATA: {message.data}")
                continue

            # we assume that the fallback confidence
            # is the same as the fallback threshold
            confidence = self.component_config[THRESHOLDS_KEY][DEFAULT_THRESHOLD_KEY]
            message.data[INTENT] = _reject_intent(confidence)
            message.data.setdefault(INTENT_RANKING_KEY, [])
            message.data[INTENT_RANKING_KEY].insert(0, _reject_intent(confidence))
            logging.info(f"MESSAGE DATA: {message.data}")

        return messages

    def _full_utterance_detected(self, message: Message):
        logging.info(f"Full Utterance detection running for: {message}.")
        logging.info(f"INTENT INFO: {message.data.get(INTENT)}")

        # is component enabled? if not, always assume full utterance
        # if not self.component_config[ENABLED_KEY]:
        #     logging.info(f"\tFull utterance because component not enabled")
        #     return True
        #
        # # is_final=True and forcing enabled?
        # if self.component_config[FORCE_FINAL_ENABLED_KEY] and message.data.get("is_final", False): #TODO: get from http parse?
        #     logging.info(f"\tFull utterance because was forced via is_final")
        #     return True
        # elif not self.component_config[FORCE_FINAL_ENABLED_KEY] and message.data.get("is_final", False): #TODO: get from http parse?
        #     logging.info("The request is trying to force a final utterance, but force_final_enabled is set to false. "
        #                  "is_final=True will be ignored. Review your config, if this is not the desired behavior.")
        #
        # # is the intent excluded from full utterance detection?
        # if self.__intent_is_excluded(message):
        #     logging.info(f"\tNOT Full utterance because intent excluded")
        #     return False
        #
        # # does it meet thresholds?
        # if not self.__is_above_thresholds(message):
        #     logging.info(f"\tNOT Full utterance because intent thresholds not reached")
        #     return False

        # does it contain the required entities?
        if self.component_config[REQUIRE_ENTITIES_ENABLED_KEY] and not self.__contains_required_entities(message):
            logging.info(f"\tNOT Full utterance because required entities not contained")
            return False
        logging.info(f"\tFull utterance because other criteria not met")
        return True

    def __is_above_thresholds(self, message: Message):
        # intent confidence is above default threshold?
        if self._nlu_confidence_below_threshold(message):
            logging.info(f"\t\tBELOW threshold for _nlu_confidence_below_threshold")
            return False
        # TODO: allow more threshold keys with intent names, so that thresholds can be set for intents.
        # TODO: this would override the default threshold

        # intent confidence is at least AMBIGUITY_THRESHOLD higher than the next highest?
        ambiguous_prediction, confidence_delta = self._nlu_prediction_ambiguous(message)
        if ambiguous_prediction:
            logging.info(f"\t\tBELOW threshold for ambiguity")
            return False

        # check if there is a requested slot and if it would be filled by the query; if not, confidence must be > self.thresholds[UNEXPECTED_UPON_REQUESTED_SLOT_KEY]
        if UNEXPECTED_UPON_REQUESTED_SLOT_KEY in self.component_config[THRESHOLDS_KEY].keys():
            #TODO: check if there is a requested slot and if it would be filled by the query; if not, confidence must be > self.thresholds[UNEXPECTED_UPON_REQUESTED_SLOT_KEY]
            print("TODO reached")
        logging.info(f"\t\tABOVE thresholds!")
        return True

    def __intent_is_excluded(self, message: Message):
        logging.info(f"Intent: {message.data.get(INTENT)[INTENT_NAME_KEY]}; excluded: { self.component_config[EXCLUDED_INTENTS_KEY]}")
        if EXCLUDED_INTENTS_KEY not in self.component_config.keys():
            return False
        intent_name = message.data.get(INTENT)[INTENT_NAME_KEY]
        if intent_name in self.component_config[EXCLUDED_INTENTS_KEY]:
            return True
        return False

    def __contains_required_entities(self, message: Message):
        #  TODO: message.data.get(INTENT) or  message.data.get(METADATA) ?
        if REQUIRE_ENTITIES_KEY not in message.data.get(INTENT).keys():
            return True

        def extract_relevant_keys(entity:dict):
            return {
                ENTITY_ATTRIBUTE_TYPE: entity.get(ENTITY_ATTRIBUTE_TYPE),
                ENTITY_ATTRIBUTE_ROLE: entity.get(ENTITY_ATTRIBUTE_ROLE),
                ENTITY_ATTRIBUTE_GROUP: entity.get(ENTITY_ATTRIBUTE_GROUP)
            }

        required_entities = [extract_relevant_keys(e) for e in message.data.get(INTENT)[REQUIRE_ENTITIES_KEY]]
        predicted_entities = [extract_relevant_keys(e) for e in  message.data.get(ENTITIES)]

        extra_entities = [e for e in predicted_entities if e not in required_entities]
        missing_entities = [e for e in required_entities if e not in predicted_entities]

        if len(extra_entities) > 0 or len(missing_entities) > 0:
            return False

        return True

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            ENABLED_KEY: DEFAULT_UTTERANCE_REJECTION_ENABLED_VALUE,
            THRESHOLDS_KEY: {
                # Reject utterances below the DEFAULT_UTTERANCE_REJECTION_THRESHOLD
                DEFAULT_THRESHOLD_KEY: DEFAULT_UTTERANCE_REJECTION_THRESHOLD,
                # If the confidence scores for the top two intent predictions are closer than `AMBIGUITY_THRESHOLD_KEY`, reject utterance
                AMBIGUITY_THRESHOLD_KEY: DEFAULT_UTTERANCE_REJECTION_AMBIGUITY_THRESHOLD
            },
            REQUIRE_ENTITIES_ENABLED_KEY: EFAULT_UTTERANCE_REJECTION_REQUIRE_ENTITIES_ENABLED_VALUE,
            # Handle rejection according to DEFAULT_UTTERANCE_REJECTION_METHOD
            METHOD_KEY: DEFAULT_UTTERANCE_REJECTION_METHOD,
            # Allow user to use is_final:True to force the NLU to accept it
            FORCE_FINAL_ENABLED_KEY: DEFAULT_UTTERANCE_REJECTION_FORCE_FINAL_VALUE,
            # List of intents to ignore and always result in rejection for
            EXCLUDED_INTENTS_KEY: DEFAULT_UTTERANCE_REJECTION_EXCLUDED_INTENTS
        }

    def _nlu_prediction_ambiguous(
        self, message: Message
    ) -> Tuple[bool, Optional[float]]:
        intents = [i for i in message.data.get(INTENT_RANKING_KEY, []) if i[INTENT_NAME_KEY] not in self.component_config[EXCLUDED_INTENTS_KEY]]
        if len(intents) >= 2:
            first_confidence = intents[0].get(PREDICTED_CONFIDENCE_KEY, 1.0)
            second_confidence = intents[1].get(PREDICTED_CONFIDENCE_KEY, 1.0)
            difference = first_confidence - second_confidence
            return (
                difference < self.component_config[THRESHOLDS_KEY][AMBIGUITY_THRESHOLD_KEY],
                difference,
            )
        return False, None

    def _nlu_confidence_below_threshold(self, message: Message) -> bool:
        nlu_confidence = message.data[INTENT].get(PREDICTED_CONFIDENCE_KEY)
        return nlu_confidence < self.component_config[THRESHOLDS_KEY][DEFAULT_THRESHOLD_KEY]

def _reject_intent(confidence: float) -> Dict[Text, Union[Text, float]]:
    return {
        INTENT_NAME_KEY: DEFAULT_UTTERANCE_REJECTION_INTENT_NAME,
        PREDICTED_CONFIDENCE_KEY: confidence,
    }