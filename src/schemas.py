from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


class LabeledRequest(BaseModel):
    id: int
    Priority: str = Field(description="One of the allowed Priority labels.")
    Service_Category: str = Field(description="One of the allowed Service Category labels.")
    Suggested_Actions: str = Field(description="One sentence, <30 words, very general, no repair steps.")


class BatchResponse(BaseModel):
    results: List[LabeledRequest]


def _get_defs_container(schema: dict) -> dict:
    return schema.get("$defs") or schema.get("definitions") or {}


def _find_labeled_request_def_key(defs: dict) -> str:
    """
    Find the key for LabeledRequest inside $defs/definitions.
    """
    if "LabeledRequest" in defs:
        return "LabeledRequest"
    for k in defs.keys():
        if k.split(".")[-1] == "LabeledRequest" or k.endswith("LabeledRequest"):
            return k
    raise KeyError("Could not find LabeledRequest in JSON Schema $defs/definitions.")


def make_response_schema(allowed_priorities: list[str], allowed_categories: list[str]) -> dict:
    """
    Create a JSON schema for Gemini structured output, injecting enums.
    """
    schema = BatchResponse.model_json_schema()

    defs = _get_defs_container(schema)
    def_key = _find_labeled_request_def_key(defs)

    labeled_schema = defs[def_key]
    props = labeled_schema.get("properties")
    if not props:
        raise KeyError(f"LabeledRequest schema for '{def_key}' has no 'properties' field.")

    # Inject enums
    props["Priority"]["enum"] = allowed_priorities
    props["Service_Category"]["enum"] = allowed_categories

    # Keep actions <30 words
    props["Suggested_Actions"]["maxLength"] = 140

    return schema