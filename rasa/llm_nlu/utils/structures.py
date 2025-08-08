from typing import (
    Text,
    Dict,
    Optional,
    Any, Union,
)

from rasa.shared.utils.io import deep_container_fingerprint, raise_warning


class GoalData:
    """  """
    def __init__(
            self,
            goals: Dict[Text, Any] = None,
    ) -> None:
        if goals is None:
            goals = {}
        self.goals = goals

    def merge(self, *others: Optional["GoalData"]) -> "GoalData":
        for o in others:
            if not o:
                continue

            for goal_name, data in o.goals.items():
                if goal_name in self.goals.keys() and self.goals[goal_name] != data:
                    raise_warning(
                        f"Found inconsistent goal definitions while merging goal data, "
                        f"overwriting {goal_name}->{self.goals[goal_name]} "
                        f"with {goal_name}->{data} during merge."
                    )

            self.goals.update(o.goals)

        return self

    def is_empty(self):
        if not self.goals:
            return True

        if len(self.goals) == 0:
            return True

        return False

    def fingerprint(self) -> Text:
        """Returns a unique hash for the goals which is stable across python runs.

        Returns:
            fingerprint of the goals
        """
        return deep_container_fingerprint(self.goals)

    def label_fingerprint(self) -> Text:
        """Fingerprints the labels in the training data.

        Returns:
            hex string as a fingerprint of the training data labels.
        """
        labels = {
            "goals": sorted(self.goals.keys()),
        }
        return deep_container_fingerprint(labels)


    def __hash__(self) -> int:
        """Calculate hash for the training data object.

        Returns:
            Hash of the training data object.
        """
        return int(self.fingerprint(), 16)

# def _validate_goals(goals: Union[Dict, List]) -> None:
#     if not isinstance(goals, dict):
#         raise InvalidDomain("Forms have to be specified as dictionary.")
#
#     for goal_name, goal_data in goals.items():
#         if goal_data is None:
#             continue
#
#         if not isinstance(goal_data, Dict):
#             raise InvalidDomain(
#                 f"The contents of goal '{goal_data}' were specified "
#                 f"as '{type(goal_data)}'. They need to be specified "
#                 f"as dictionary. Please see {DOCS_URL_FORMS} "
#                 f"for more information."
#             )
#
#         # TODO: Validate:
#         # - consists of goals with description and steps
#         # - steps can be "collect" or "action"
#         # - step collect: