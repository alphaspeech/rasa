import os
from pathlib import Path
import copy
from typing import Callable, List, Optional, Text, Dict, Any

from _pytest.monkeypatch import MonkeyPatch
import pytest


from rasa.engine.graph import ExecutionContext, GraphComponent, GraphSchema, SchemaNode
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.validators.finetuning_validator import FinetuningValidator
from rasa.graph_components.validators.llm_extension_validator import LLMExtensionValidator
from rasa.llm_nlu.utils.structures import GoalData
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.core.policies.rule_policy import RulePolicy
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH, DEFAULT_GOALS_PATH,
)
from rasa.shared.core.domain import KEY_RESPONSES, Domain
from rasa.shared.importers.goals import ExtendedRasaFileImporter
import rasa.shared.utils.io
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.constants import ACTION_NAME, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


@pytest.fixture
def default_resource() -> Resource:
    return Resource("LLMExtensionValidator")


ValidationMethodType = Callable[
    [ExtendedRasaFileImporter, Dict[Text, Any]], ExtendedRasaFileImporter
]


@pytest.fixture
def get_llm_extension_validator(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    default_resource: Resource,
) -> Callable[[bool, bool, Dict[Text, Any], GraphSchema], LLMExtensionValidator]:
    def inner(
        load: bool,
        config: Dict[Text, Any],
        graph_schema: Optional[GraphSchema] = None,
    ) -> LLMExtensionValidator:
        if load:
            constructor = LLMExtensionValidator.load
        else:
            constructor = LLMExtensionValidator.create
        if graph_schema is not None:
            default_execution_context.graph_schema = graph_schema
        return constructor(
            config={**LLMExtensionValidator.get_default_config(), **config},
            execution_context=default_execution_context,
            model_storage=default_model_storage,
            resource=default_resource,
        )

    return inner


@pytest.fixture
def get_validation_method(
    get_llm_extension_validator: Callable[[bool, bool], LLMExtensionValidator]
) -> Callable[[bool, bool, bool, bool, GraphSchema], ValidationMethodType]:
    def inner(
        load: bool,
        graph_schema: Optional[GraphSchema] = None,
    ) -> ValidationMethodType:
        validator = get_llm_extension_validator(
            load=load,
            config={},
            graph_schema=graph_schema,
        )

        return validator.validate

    return inner


class DummyGoalDataImporter(ExtendedRasaFileImporter):
    def __init__(self, goals: Dict[Text, Any]) -> None:
        self.goal_data = GoalData(goals)

    def get_config(self) -> Dict:
        return {}

    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        return TrainingData()

    def get_goals(self) -> GoalData:
        return self.goal_data


class EmptyDataImporter(DummyGoalDataImporter):
    def __init__(self) -> None:
        super().__init__([])


def _project_files(
    project: Text,
    config_file: Text = DEFAULT_CONFIG_PATH,
    domain: Text = DEFAULT_DOMAIN_PATH,
    goals_data_paths: Text = DEFAULT_GOALS_PATH,
) -> ExtendedRasaFileImporter:
    paths = {
        "config_file": config_file,
        "domain_path": domain,
        "training_data_paths": None,
        "goals_data_paths": goals_data_paths,
    }
    paths = {
        k: v if v is None or Path(v).is_absolute() else os.path.join(project, v)
        for k, v in paths.items()
    }
    paths["goals_data_paths"] = [paths["goals_data_paths"]]

    return ExtendedRasaFileImporter(**paths)

def _get_example_schema(num_epochs: int = 5, other_parameter: int = 10) -> GraphSchema:
    example_configs = [
        {
            "epochs": num_epochs,
            "other-parameter": other_parameter,
            "some-parameter": "bla",
        },
        {"epochs": num_epochs, "yet-other-parameter": 344},
        {"no-epochs-defined-here": None},
    ]
    return GraphSchema(
        nodes={
            f"node-{idx}": SchemaNode(
                needs={}, uses=GraphComponent, constructor_name="", fn="", config=config
            )
            for idx, config in enumerate(example_configs)
        }
    )


def test_validate_after_changing_epochs_in_config(
    get_validation_method: Callable[..., ValidationMethodType]
):
    # training
    schema1 = _get_example_schema(num_epochs=5)
    validate = get_validation_method(
        load=False, graph_schema=schema1
    )
    validate(importer=EmptyDataImporter())

    # change schema - replace all epoch settings by a different value
    schema2 = _get_example_schema(num_epochs=5)
    for node in schema2.nodes.values():
        node.constructor_name = "other"

    # finetuning - does not complain
    loaded_validate = get_validation_method(
        load=True, graph_schema=schema2
    )
    loaded_validate(importer=EmptyDataImporter())


def test_validate_after_changing_constructor(
    get_validation_method: Callable[..., ValidationMethodType]
):
    # training
    schema1 = _get_example_schema(num_epochs=5)
    validate = get_validation_method(
        load=False, graph_schema=schema1
    )
    validate(importer=EmptyDataImporter())

    # change schema - replace all epoch settings by a different value
    schema2 = _get_example_schema(num_epochs=10)

    # finetuning - does not complain
    loaded_validate = get_validation_method(
        load=True, graph_schema=schema2
    )
    loaded_validate(importer=EmptyDataImporter())


def test_validate_after_removing_node_from_schema(
    get_validation_method: Callable[..., ValidationMethodType]
):
    # training
    schema1 = _get_example_schema(num_epochs=5)
    validate = get_validation_method(
        load=False, graph_schema=schema1
    )
    validate(importer=EmptyDataImporter())

    # change schema - remove a node
    schema2 = copy.deepcopy(schema1)
    schema2.nodes.pop(next(iter(schema2.nodes.keys())))

    # finetuning raises - doesn't matter if it's nlu/core/both
    loaded_validate = get_validation_method(
        load=True, graph_schema=schema2
    )
    with pytest.raises(InvalidConfigException):
        loaded_validate(importer=EmptyDataImporter())


def test_validate_after_adding_node_to_schema(
    get_validation_method: Callable[..., ValidationMethodType]
):
    # training
    schema1 = _get_example_schema()
    schema2 = copy.deepcopy(schema1)
    schema2.nodes.pop(next(iter(schema2.nodes.keys())))

    validate = get_validation_method(
        load=False, graph_schema=schema2
    )
    validate(importer=EmptyDataImporter())

    # change schema - continue with the schema with one more node than before
    assert len(schema1.nodes) > len(schema2.nodes)

    # finetuning raises -  doesn't matter if it's nlu/core/both
    loaded_validate = get_validation_method(
        load=True, graph_schema=schema1
    )
    with pytest.raises(InvalidConfigException):
        loaded_validate(importer=EmptyDataImporter())


@pytest.mark.parametrize(
    "what",
    [
        (what)
        for what in ["uses", "needs", "fn", "config"]
    ],
)
def test_validate_after_replacing_something_in_schema(
    get_validation_method: Callable[..., ValidationMethodType],
    what: Text,
):
    # training
    schema1 = _get_example_schema()
    validate = get_validation_method(
        load=False, graph_schema=schema1
    )
    validate(importer=EmptyDataImporter())

    # change schema
    schema2 = copy.deepcopy(schema1)
    schema_node = schema2.nodes["node-0"]
    if what == "uses":
        schema_node.uses = WhitespaceTokenizer
    elif what == "fn":
        schema_node.fn = "a-new-function"
    elif what == "needs":
        schema_node.needs = {"something-new": "node-1"}
    elif what == "config":
        schema_node.config["other-parameter"] = "some-new-value"
    else:
        assert False, "Please fix this test."

    # finetuning raises -  doesn't matter if it's nlu/core/both
    loaded_validate = get_validation_method(
        load=True, graph_schema=schema2
    )
    with pytest.raises(InvalidConfigException):
        loaded_validate(importer=EmptyDataImporter())


def test_validate_after_adding_adding_default_parameter(
    get_validation_method: Callable[..., ValidationMethodType]
):
    # create a schema and rely on rasa to fill in defaults later
    schema1 = _get_example_schema()
    schema1.nodes["nlu-node"] = SchemaNode(
        needs={}, uses=WhitespaceTokenizer, constructor_name="", fn="", config={}
    )
    schema1.nodes["core-node"] = SchemaNode(
        needs={}, uses=RulePolicy, constructor_name="", fn="", config={}
    )

    # training
    validate = get_validation_method(
        load=False, graph_schema=schema1
    )
    validate(importer=EmptyDataImporter())

    # same schema -- we just explicitly pass default values
    schema2 = copy.deepcopy(schema1)
    schema2.nodes["nlu-node"] = SchemaNode(
        needs={},
        uses=WhitespaceTokenizer,
        constructor_name="",
        fn="",
        config=WhitespaceTokenizer.get_default_config(),
    )
    schema2.nodes["core-node"] = SchemaNode(
        needs={},
        uses=RulePolicy,
        constructor_name="",
        fn="",
        config=RulePolicy.get_default_config(),
    )

    # finetuning *does not raise*
    loaded_validate = get_validation_method(
        load=True, graph_schema=schema2
    )
    loaded_validate(importer=EmptyDataImporter())


@pytest.mark.parametrize(
    "min_compatible_version, old_version, can_tune",
    [
        (old_version, min_compatible_version, can_tune)
        for old_version, min_compatible_version, can_tune in [
            ("2.1.0", "2.1.0", True),
            ("2.0.0", "2.1.0", True),
            ("2.1.0", "2.0.0", False),
        ]
    ],
)
def test_validate_with_other_version(
    monkeypatch: MonkeyPatch,
    get_validation_method: Callable[..., ValidationMethodType],
    min_compatible_version: Text,
    old_version: Text,
    can_tune: bool,
):
    monkeypatch.setattr(rasa, "__version__", old_version)
    monkeypatch.setattr(
        rasa.graph_components.validators.finetuning_validator,
        "MINIMUM_COMPATIBLE_VERSION",
        min_compatible_version,
    )

    # training
    importer = DummyGoalDataImporter([Message(data={INTENT: "dummy"})])
    validate = get_validation_method(load=False)
    validate(importer=importer)

    # finetuning
    validate = get_validation_method(load=True)
    if not can_tune:
        with pytest.raises(InvalidConfigException):
            validate(importer=importer)
    else:
        validate(importer=importer)


def test_loading_without_persisting(
    get_finetuning_validator: Callable[
        [bool, bool, Dict[Text, bool]], FinetuningValidator
    ]
):
    with pytest.raises(ValueError):
        get_finetuning_validator(finetuning=False, load=True, config={})
