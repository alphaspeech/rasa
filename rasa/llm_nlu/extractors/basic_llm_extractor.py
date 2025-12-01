from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Text, Type

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
import rasa.shared.utils.io
from rasa.graph_components.providers.domain_provider import DomainProvider
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
)
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.llm_nlu.utils.send_request import fill_prompt_template, send_request, get_simplified_history, \
    create_slot_extraction_model

logger = logging.getLogger(__name__)

MODEL = "model"

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=True
)
class LLMEntityExtractor(DomainProvider, EntityExtractorMixin):
    """Extracts entities via lookup tables and regexes defined in the training data."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return []

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            MODEL: "qwen2.5:14b",
        }

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> LLMEntityExtractor:
        """Creates a new `GraphComponent`.

        Args:
            config: This config overrides the `default_config`.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run. Unused.

        Returns: An instantiated `GraphComponent`.
        """
        return cls(config, model_storage, resource)

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        """Creates a new instance.

        Args:
            config: The configuration.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            patterns: a list of patterns
        """
        # graph component
        self._config = {**self.get_default_config(), **config}
        self._model_storage = model_storage
        self._resource = resource

    def train(self, training_data: TrainingData) -> Resource:
        """Extract patterns from the training data.

        Args:
            training_data: the training data
        """
        if not training_data.entity_examples:
            logger.debug(
                "No training examples with entities present. Skip training"
                "of 'CRFEntityExtractor'."
            )
            return self._resource

        if not self.SlotExtraction:
            rasa.shared.utils.io.raise_warning(
                "No lookup tables or regexes defined in the training data that have "
                "a name equal to any entity in the training data. In order for this "
                "component to work you need to define valid lookup tables or regexes "
                "in the training data."
            )

        self.persist()
        return self._resource

    def process(self, messages: List[Message], domain: Optional[Domain] = None) -> List[Message]:
        """Extracts entities from messages and appends them to the attribute.

        If no patterns where found during training, then the given messages will not
        be modified. In particular, if no `ENTITIES` attribute exists yet, then
        it will *not* be created.

        If no pattern can be found in the given message, then no entities will be
        added to any existing list of entities. However, if no `ENTITIES` attribute
        exists yet, then an `ENTITIES` attribute will be created.

        Returns:
           the given list of messages that have been modified
        """
        slot_extraction_model = domain.LLMSlotExtraction

        for message in messages:
            extracted_entities = self._extract_entities(message, slot_extraction_model)
            extracted_entities = self.add_extractor_name(extracted_entities)
            message.set(
                ENTITIES,
                message.get(ENTITIES, []) + extracted_entities,
                add_to_output=True,
            )

        return messages

    def _extract_entities(self, message: Message, slot_extraction_model) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message.

        Args:
            message: a message
        Returns:
            a list of dictionaries describing the entities
        """
        entities = []

        prompt = fill_prompt_template(
            SLOT_MAPPING_TEMPLATE,
            user_input=message.get(TEXT),
            slot_state=slot_extraction_model.model_json_schema(), # self.slot_state, # TODO: Get actual slot state
            #conversation_history=self.conversation_history # TODO: determine if needed
        )
        new_slot_state = send_request(prompt, format=slot_extraction_model.model_json_schema(), model=self._config.get("model"))
        logger.info(f"The new slot state is: {new_slot_state}")
        self.slot_state = new_slot_state

        start_index = "unknown"
        end_index = "unknown"
        for key, value in new_slot_state.items():
            if value is None:
                continue
            entities.append(
                {
                    ENTITY_ATTRIBUTE_TYPE: key,
                    ENTITY_ATTRIBUTE_START: start_index,
                    ENTITY_ATTRIBUTE_END: end_index,
                    ENTITY_ATTRIBUTE_VALUE: value
                }
            )
        logger.info(f"Predicted entities: {entities}")
        return entities

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> LLMEntityExtractor:
        """Loads trained component (see parent class for full docstring)."""
        return cls(
            config,
            model_storage=model_storage,
            resource=resource,
        )

    def persist(self) -> None:
        """Persist this model."""
        pass


SLOT_MAPPING_TEMPLATE = """
    Your job is to update the JSON containing the structured information gathered from the dialogue based on the lastest user input.
    If the information could be directly found in the user query, set the certainty metric to 1.0.
    If you had to use your best judgement to set the slot, set a value between 0.0 and 1.0 for the certainty metric. Assume that anything below 0.7 means asking further would make sense.
    Do not change certainty values for information that isn't the topic of the current user query.

    Here is the current information as a json:
    {slot_state}

    The latest user input was:
    {user_input}

    Now update the JSON containing the information based on the lastest user input.
"""

#     Consider the conversation history. "user" is the user, "bot" is your replies. The conversation history is:
#     {conversation_history}