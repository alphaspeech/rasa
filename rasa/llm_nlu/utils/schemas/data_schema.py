from typing import Dict, Text, Any

#TODO: Consider how more actions could be added dynamically whenever created somewhere else


def rasa_goals_data_schema() -> Dict[Text, Any]:
    """Returns: schema of the Rasa NLU data format (json format)."""
    COLLECT_ACTION_SCHEMA = {
        "properties": {
            "collect": {"type": "string"},
            "description": {"type": "string"}
        },
        "required": ["collect", "description"],
        "additionalProperties": False
    }

    RUN_ACTION_SCHEMA = {
        "properties": {
            "action": {"type": "string"}
        },
        "required": ["action"],
        "additionalProperties": False
    }

    step_schema = {
        "type": "array",
        "items": {
          "type": "object",
          "oneOf": [
            COLLECT_ACTION_SCHEMA,
            RUN_ACTION_SCHEMA
          ]
        }
    }

    goal_schema = {
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z0-9_]+$": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Explanation of when this goal applies"
                    },
                    "steps": step_schema
                },
              "required": ["description", "steps"],
              "additionalProperties": False
            }
        }
    }

    return {
        "type": "object",
        "properties": {
            "goals": {
                "type": "object",
                "patternProperties": {
                    "^[a-zA-Z0-9_]+$": goal_schema
                }
            }
        },
        "additionalProperties": False,
    }
