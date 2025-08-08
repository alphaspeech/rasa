from __future__ import annotations
import copy
import logging
from rasa.nlu.featurizers.featurizer import Featurizer

from typing import Any, Dict, Optional, Text, List, Type
import requests
import json

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.nlu.training_data import util
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.classifiers.diet_classifier import (
    LABEL_KEY,
    LABEL_SUB_KEY,
    DIETClassifier,
)
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.utils.tensorflow.constants import (
    RETRIEVAL_INTENT,
    USE_TEXT_AS_LABEL,
)
from rasa.nlu.constants import (
    RESPONSE_SELECTOR_PROPERTY_NAME,
    RESPONSE_SELECTOR_RETRIEVAL_INTENTS,
    RESPONSE_SELECTOR_RESPONSES_KEY,
    RESPONSE_SELECTOR_PREDICTION_KEY,
    RESPONSE_SELECTOR_RANKING_KEY,
    RESPONSE_SELECTOR_UTTER_ACTION_KEY,
    RESPONSE_SELECTOR_DEFAULT_INTENT,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    RESPONSE,
    INTENT_RESPONSE_KEY,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)

ENDPOINT = "endpoint"
MODEL = "model"
MAX_CHAR = "max_char"
STREAM = "stream"
TEXT = "text"

from rasa.utils.tensorflow.model_data import RasaModelData
from rasa.utils.tensorflow.models import RasaModel

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class AlphaspeechOllamaResponder(GraphComponent):
    """Response generator using Alphaspeech Ollama endpoint.

    The response generator uses user inputs
    to generate a reply end-to-end.

    When preceeded by a slot extractor, it should be able to use slot information.
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return []

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            ENDPOINT: "https://llm.dev.alphaspeech.de/api/chat",
            MODEL: "linguwerk",
            MAX_CHAR: 150,
            STREAM: False,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        index_label_id_mapping: Optional[Dict[int, Text]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        model: Optional[RasaModel] = None,
        all_retrieval_intents: Optional[List[Text]] = None,
        responses: Optional[Dict[Text, List[Dict[Text, Any]]]] = None,
        sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None,
    ) -> None:
        """Declare instance variables with default values.

        Args:
            config: Configuration for the component.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.
            index_label_id_mapping: Mapping between label and index used for encoding.
            entity_tag_specs: Format specification all entity tags.
            model: Model architecture.
            all_retrieval_intents: All retrieval intents defined in the data.
            responses: All responses defined in the data.
            finetune_mode: If `True` loads the model with pre-trained weights,
                otherwise initializes it with random weights.
            sparse_feature_sizes: Sizes of the sparse features the model was trained on.
        """
        self.component_config = config
        self.context_storage = {}
        super().__init__(
            config,
            model_storage,
            resource,
            execution_context,
            index_label_id_mapping,
            entity_tag_specs,
            model,
            sparse_feature_sizes=sparse_feature_sizes,
        )


    def _set_message_property(
        self, message: Message, prediction_dict: Dict[Text, Any], selector_key: Text
    ) -> None:
        message_selector_properties = message.get(RESPONSE_SELECTOR_PROPERTY_NAME, {})
        message_selector_properties[
            RESPONSE_SELECTOR_RETRIEVAL_INTENTS
        ] = self.all_retrieval_intents
        message_selector_properties[selector_key] = prediction_dict
        message.set(
            RESPONSE_SELECTOR_PROPERTY_NAME,
            message_selector_properties,
            add_to_output=True,
        )

    def _send_request(self, sender_id:str, user_input:str):
        headers = {
            "Content-Type": "application/json"
        }

        data = {"model": self.component_config.get(MODEL), "stream": False}

        if sender_id not in self.context_storage.keys():
            self.context_storage[sender_id] = []
        self.context_storage[sender_id].append({"role": "user", "content": user_input})
        data["messages"] = self.context_storage[sender_id]
        response = requests.post(self.component_config.get(ENDPOINT), headers=headers, data=json.dumps(data))
        content = json.loads(response.content.decode('utf-8'))
        self.context_storage[sender_id].append(content["message"])
        reply = content["message"]["content"]

        return reply

    def process(self, messages: List[Message]) -> List[Message]:
        """Selects most like response for message.

        Args:
            messages: List containing latest user message.

        Returns:
            List containing the message augmented with the most likely response,
            the associated intent_response_key and its similarity to the input.
        """
        full_message_text = ""
        for message in messages:
            logging.info(message)
            full_message_text = full_message_text + ". " + message

        sender_id = ""
        out = self._send_request(sender_id, user_input=full_message_text)

        logger.debug(
            f"LLM generated the following e2e responde: {out}"
        )

        prediction_dict = {
            RESPONSE_SELECTOR_PREDICTION_KEY: {
                TEXT: out,
                MODEL: self.component_config.get(MODEL),
            }
        }

        self._set_message_property(messages[-1], prediction_dict, RESPONSE_SELECTOR_DEFAULT_INTENT)


        return messages



    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> AlphaspeechOllamaResponder:
        """Loads the trained model from the provided directory."""
        model = super().load(
            config, model_storage, resource, execution_context, **kwargs
        )

        try:
            with model_storage.read_from(resource) as model_path:
                file_name = cls.__name__
                responses = rasa.shared.utils.io.read_json_file(
                    model_path / f"{file_name}.responses.json"
                )
                all_retrieval_intents = rasa.shared.utils.io.read_json_file(
                    model_path / f"{file_name}.retrieval_intents.json"
                )
                model.responses = responses
                model.all_retrieval_intents = all_retrieval_intents
                return model
        except ValueError:
            logger.debug(
                f"Failed to load {cls.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls(config, model_storage, resource, execution_context)

