from typing import List

from rasa.core.channels import UserMessage
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.importers.goals import ExtendedRasaFileImporter

PLACEHOLDER_IMPORTER = "__importer__"
PLACEHOLDER_MESSAGE = "__message__"
PLACEHOLDER_TRACKER = "__tracker__"
RESERVED_PLACEHOLDERS = {
    PLACEHOLDER_IMPORTER: ExtendedRasaFileImporter,
    PLACEHOLDER_MESSAGE: List[UserMessage],
    PLACEHOLDER_TRACKER: DialogueStateTracker,
}
