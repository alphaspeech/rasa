import os
from typing import Text

import pytest
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.goals_provider import GoalsProvider
from rasa.llm_nlu.utils.structures import GoalData
from rasa.shared.importers.goals import ExtendedRasaFileImporter


def test_goals_provider(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config_path: Text,
    goals_path: Text,
):
    resource = Resource("xy")
    importer = ExtendedRasaFileImporter.load_from_config(config_path, goals_data_paths=[goals_path])

    component = GoalsProvider.create(
        GoalsProvider.get_default_config(),
        default_model_storage,
        resource,
        default_execution_context,
    )
    assert isinstance(component, GoalsProvider)

    provided_goals = component.provide(importer)
    compare_goals = importer.get_goals()

    assert isinstance(provided_goals, GoalData)
    assert provided_goals.fingerprint() == compare_goals.fingerprint()
    assert not provided_goals.is_empty()
