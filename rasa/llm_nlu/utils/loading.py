import os
from typing import (
    Text,
)

from rasa.llm_nlu.utils.structures import GoalData
from rasa.shared.core.domain import KEY_GOALS
from rasa.shared.utils.io import list_files, read_yaml_file


def load_goals_from_file(
    goal_file: Text,
) -> GoalData:
    """Loads core training data from the specified files.

    Args:
        goal_file: File or directory containing goals.

    Returns:
        Goal Data containing loaded goals
    """
    if not os.path.exists(goal_file):
        raise ValueError(f"File '{goal_file}' does not exist.")

    if not os.path.isfile(goal_file): # then is directory
        files = list_files(goal_file)
        goal_data_sets = [load_goals_from_file(f) for f in files]
        return GoalData().merge(*goal_data_sets)

    content = read_yaml_file(goal_file)

    return GoalData(content[KEY_GOALS])