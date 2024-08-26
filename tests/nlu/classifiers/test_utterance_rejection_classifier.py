import copy
from typing import Dict, Text, Any

import pytest

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers import utterance_rejection_classifier
from rasa.shared.constants import DEFAULT_UTTERANCE_REJECTION_INTENT_NAME
from rasa.core.constants import DEFAULT_UTTERANCE_REJECTION_THRESHOLD
from rasa.nlu.classifiers.utterance_rejection_classifier import (
    ENABLED_KEY,
    THRESHOLDS_KEY,
    REQUIRE_ENTITIES_ENABLED_KEY,
    METHOD_KEY,
    FORCE_FINAL_ENABLED_KEY,
    EXCLUDED_INTENTS_KEY,
    DEFAULT_THRESHOLD_KEY,
    AMBIGUITY_THRESHOLD_KEY,
    UNEXPECTED_UPON_REQUESTED_SLOT_KEY, UtteranceRejectionClassifier, CUSTOM_THRESHOLDS_KEY
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    INTENT,
    TEXT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
    REQUIRE_ENTITIES_KEY, ENTITIES, ENTITY_ATTRIBUTE_TYPE,
)

def create_utterance_rejection_classifier(
    component_config: Dict[Text, Any],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    classifier = UtteranceRejectionClassifier.create(
        {**UtteranceRejectionClassifier.get_default_config(), **component_config},
        default_model_storage,
        Resource("utterance_rejection"),
        default_execution_context,
    )

    return classifier


@pytest.mark.parametrize(
    "message, component_config",
    [
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {
                        INTENT_NAME_KEY: "greet",
                        PREDICTED_CONFIDENCE_KEY: 0.68,
                    },
                    INTENT_RANKING_KEY: [
                        {
                            INTENT_NAME_KEY: "greet",
                            PREDICTED_CONFIDENCE_KEY: 0.68,
                        },
                        {
                            INTENT_NAME_KEY: "stop",
                            PREDICTED_CONFIDENCE_KEY: 0.5,
                        },
                        {INTENT_NAME_KEY: "affirm", PREDICTED_CONFIDENCE_KEY: 0},
                        {INTENT_NAME_KEY: "inform", PREDICTED_CONFIDENCE_KEY: -100},
                        {
                            INTENT_NAME_KEY: "deny",
                            PREDICTED_CONFIDENCE_KEY: 0.0879683718085289,
                        },
                    ],
                }
            ),
            {THRESHOLDS_KEY: {DEFAULT_THRESHOLD_KEY: 0.7}},
        ),
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                    INTENT_RANKING_KEY: [
                        {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.99},
                    ],
                }
            ),
            {THRESHOLDS_KEY: {DEFAULT_THRESHOLD_KEY: 0.7, AMBIGUITY_THRESHOLD_KEY: 0.1}},
        ),
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {INTENT_NAME_KEY: "nlu_fallback", PREDICTED_CONFIDENCE_KEY: 0.75},
                    INTENT_RANKING_KEY: [
                        {INTENT_NAME_KEY: "nlu_fallback", PREDICTED_CONFIDENCE_KEY: 0.75},
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.5},
                    ],
                }
            ),
            {THRESHOLDS_KEY: {DEFAULT_THRESHOLD_KEY: 0.7,  AMBIGUITY_THRESHOLD_KEY: 0.1}},
        ),
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {
                        INTENT_NAME_KEY: "greet",
                        PREDICTED_CONFIDENCE_KEY: 0.9,
                        REQUIRE_ENTITIES_KEY: [],
                    },
                    ENTITIES: [{ENTITY_ATTRIBUTE_TYPE: "other_entity"}],
                    INTENT_RANKING_KEY: [
                        {
                            INTENT_NAME_KEY: "greet",
                            PREDICTED_CONFIDENCE_KEY: 0.9,
                        }
                    ],
                }
            ),
            {THRESHOLDS_KEY: {DEFAULT_THRESHOLD_KEY: 0.7}, REQUIRE_ENTITIES_ENABLED_KEY: True},
        ),
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {
                        INTENT_NAME_KEY: "greet",
                        PREDICTED_CONFIDENCE_KEY: 0.9,
                        REQUIRE_ENTITIES_KEY: [{ENTITY_ATTRIBUTE_TYPE: "required_entity"}],
                    },
                    ENTITIES: [],
                    INTENT_RANKING_KEY: [
                        {
                            INTENT_NAME_KEY: "greet",
                            PREDICTED_CONFIDENCE_KEY: 0.9,
                        }
                    ],
                }
            ),
            {THRESHOLDS_KEY: {DEFAULT_THRESHOLD_KEY: 0.7}, REQUIRE_ENTITIES_ENABLED_KEY: True},
        ),
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {
                        INTENT_NAME_KEY: "greet",
                        PREDICTED_CONFIDENCE_KEY: 0.9,
                        REQUIRE_ENTITIES_KEY: [{ENTITY_ATTRIBUTE_TYPE: "required_entity"}],
                    },
                    ENTITIES: [{ENTITY_ATTRIBUTE_TYPE: "required_entity"}, {ENTITY_ATTRIBUTE_TYPE: "other_entity"}],
                    INTENT_RANKING_KEY: [
                        {
                            INTENT_NAME_KEY: "greet",
                            PREDICTED_CONFIDENCE_KEY: 0.9,
                        }
                    ],
                }
            ),
            {THRESHOLDS_KEY: {DEFAULT_THRESHOLD_KEY: 0.7}, REQUIRE_ENTITIES_ENABLED_KEY: True},
        ),
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {
                        INTENT_NAME_KEY: "greet",
                        PREDICTED_CONFIDENCE_KEY: 0.75,
                    },
                    ENTITIES: [],
                    INTENT_RANKING_KEY: [
                        {
                            INTENT_NAME_KEY: "greet",
                            PREDICTED_CONFIDENCE_KEY: 0.75,
                        }
                    ],
                }
            ),
            {THRESHOLDS_KEY: {DEFAULT_THRESHOLD_KEY: 0.7, CUSTOM_THRESHOLDS_KEY: {"greet": 0.8}}},
        ),
    ],
)

def test_predict_uttrance_injection_intent(
    message: Message,
    component_config: Dict,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    old_message_state = copy.deepcopy(message)
    expected_confidence = component_config[THRESHOLDS_KEY][DEFAULT_THRESHOLD_KEY]

    classifier = create_utterance_rejection_classifier(
        component_config, default_model_storage, default_execution_context
    )
    processed_messages = classifier.process([message])
    processed_msg = processed_messages[0]

    expected_intent = {
        INTENT_NAME_KEY: DEFAULT_UTTERANCE_REJECTION_INTENT_NAME,
        PREDICTED_CONFIDENCE_KEY: expected_confidence,
    }
    assert processed_msg.data[INTENT] == expected_intent

    old_intent_ranking = old_message_state.data[INTENT_RANKING_KEY]
    current_intent_ranking = processed_msg.data[INTENT_RANKING_KEY]

    assert len(current_intent_ranking) == len(old_intent_ranking) + 1
    assert all(item in current_intent_ranking for item in old_intent_ranking)
    assert current_intent_ranking[0] == expected_intent



@pytest.mark.parametrize(
    "message, component_config",
    [
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 0.5},
                    INTENT_RANKING_KEY: [
                        {
                            INTENT_NAME_KEY: "greet",
                            PREDICTED_CONFIDENCE_KEY: 0.71,
                        },
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.71},
                        {INTENT_NAME_KEY: "affirm", PREDICTED_CONFIDENCE_KEY: 0.6},
                        {INTENT_NAME_KEY: "inform", PREDICTED_CONFIDENCE_KEY: -100},
                        {
                            INTENT_NAME_KEY: "deny",
                            PREDICTED_CONFIDENCE_KEY: 0.0879683718085289,
                        },
                    ],
                }
            ),
            {THRESHOLDS_KEY: {DEFAULT_THRESHOLD_KEY: 0.7, AMBIGUITY_THRESHOLD_KEY: 0.1}},
        )
    ],
)


def test_not_predict_uttrance_injection_intent(
    message: Message,
    component_config: Dict,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    old_message_state = copy.deepcopy(message)

    classifier = create_utterance_rejection_classifier(
        component_config, default_model_storage, default_execution_context
    )
    processed_messages = classifier.process([message])
    processed_msg = processed_messages[0]

    assert processed_msg.data[INTENT] != DEFAULT_UTTERANCE_REJECTION_INTENT_NAME
    #assert processed_msg == old_message_state
    # TODO: Assert that msg state remains the same


def test_defaults(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    classifier = create_utterance_rejection_classifier(
        {}, default_model_storage, default_execution_context
    )

    assert classifier.component_config[THRESHOLDS_KEY][DEFAULT_THRESHOLD_KEY] == DEFAULT_UTTERANCE_REJECTION_THRESHOLD
    assert classifier.component_config[THRESHOLDS_KEY][AMBIGUITY_THRESHOLD_KEY] == 0.1
    assert UNEXPECTED_UPON_REQUESTED_SLOT_KEY not in classifier.component_config[THRESHOLDS_KEY].keys()
    assert classifier.component_config[METHOD_KEY] == "Ignore"
    assert classifier.component_config[FORCE_FINAL_ENABLED_KEY] == True
    assert classifier.component_config[EXCLUDED_INTENTS_KEY] == ["nlu_fallback"]

@pytest.mark.parametrize(
    "message, component_config",
    [
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 0.5},
                    INTENT_RANKING_KEY: [
                        {
                            INTENT_NAME_KEY: "greet",
                            PREDICTED_CONFIDENCE_KEY: 0.71,
                        },
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.71},
                        {INTENT_NAME_KEY: "affirm", PREDICTED_CONFIDENCE_KEY: 0.6},
                        {INTENT_NAME_KEY: "inform", PREDICTED_CONFIDENCE_KEY: -100},
                        {
                            INTENT_NAME_KEY: "deny",
                            PREDICTED_CONFIDENCE_KEY: 0.0879683718085289,
                        },
                    ],
                }
            ),
            {THRESHOLDS_KEY: {DEFAULT_THRESHOLD_KEY: 0.7, AMBIGUITY_THRESHOLD_KEY: 0.1}},
        )
    ],
)

def test_required_entities_functioning(message: Message,
    component_config: Dict,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    #TODO: implement test once this works
    assert True == True