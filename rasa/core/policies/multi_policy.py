import abc
from typing import (
    Any,
    List,
    Optional,
    Text,
    Dict,
)

from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker


class MultiPolicy(Policy):
    """Common parent class for policies that can return multiple predictions."""

    @abc.abstractmethod
    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> List[PolicyPrediction]:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: The tracker containing the conversation history up to now.
            domain: The model's domain.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to make predictions.

        Returns:
             The list of predictions.
        """
        raise NotImplementedError("Policy must have the capacity to predict.")
