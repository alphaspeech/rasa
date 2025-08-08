from __future__ import annotations

import logging
from typing import Dict, Text, Any, Optional

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.llm_nlu.utils.loading import load_goals_from_file
from rasa.llm_nlu.utils.structures import GoalData
from rasa.shared.constants import DEFAULT_GOALS_PATH
from rasa.shared.core.domain import Domain
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.importers.goals import ExtendedRasaFileImporter
from rasa.shared.importers.importer import TrainingDataImporter


class GoalsProvider(GraphComponent):
    """Provides domain during training and inference time."""

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        goals: Optional[GoalData] = None,
    ) -> None:
        """Creates goals provider."""
        self._config = config
        self._model_storage = model_storage
        self._resource = resource
        self._goals = goals
        logging.info("Successful init of GoalsProvider")

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GoalsProvider:
        """Creates component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> GoalsProvider:
        """Creates provider using a persisted version of itself."""
        with model_storage.read_from(resource) as resource_directory:
            goals = load_goals_from_file(resource_directory)
        return cls(model_storage, resource, goals)

    def provide(self, importer: ExtendedRasaFileImporter) -> GoalData:
        """Provides goals from training data during training."""
        goals = importer.get_goals()
        logging.info("GOALS SUCCESSFUL??????????????????????????????")
        logging.info(self._goals)
        return goals

    # def provide_inference(self) -> GoalData:
    #     """Provides the goals during inference."""
    #     if self._goals is None:
    #         # This can't really happen but if it happens then we fail early
    #         raise InvalidConfigException(
    #             "No goals were found. This is required for "
    #             "making model predictions. Please make sure to "
    #             "provide a valid goals during training."
    #         )
    #     return self._goals
