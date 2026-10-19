# Harbor ATIF models (Apache-2.0; see LICENSE).
# Source: harbor-framework/harbor@357c5e74020fc0890d69fb3a7db610896b9ce57d
# Modified: local imports; trajectory schema version pinned to v1.8.
"""Agent configuration model for ATIF trajectories."""

from typing import Any

from pydantic import BaseModel, Field


class Agent(BaseModel):
    """Agent configuration."""

    name: str = Field(
        default=...,
        description="The name of the agent system",
    )
    version: str = Field(
        default=...,
        description="The version identifier of the agent system",
    )
    model_name: str | None = Field(
        default=None,
        description="Default LLM model used for this trajectory",
    )
    tool_definitions: list[dict[str, Any]] | None = Field(
        default=None,
        description="Array of tool/function definitions available to the agent. Each element follows OpenAI's function calling schema.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom agent configuration details",
    )

    model_config = {"extra": "forbid"}
