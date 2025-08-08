from typing import Optional, Text, List, Union

from rasa.llm_nlu.utils.structures import GoalData
from rasa.shared.data import get_data_files
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.importers.utils import goals_data_from_paths

class ExtendedRasaFileImporter(RasaFileImporter):
    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
        goals_data_paths: Optional[Union[List[Text], Text]] = None,
    ):
        super().__init__(config_file, domain_path, training_data_paths)

        self._goals_files = get_data_files(
            goals_data_paths, YAMLStoryReader.is_goals_file
        )

    def get_goals(self) -> GoalData:
        return goals_data_from_paths(self._goals_files)