# Harbor ATIF models (Apache-2.0; see LICENSE).
# Source: harbor-framework/harbor@357c5e74020fc0890d69fb3a7db610896b9ce57d
# Modified: local imports; trajectory schema version pinned to v1.8.
"""Observation model for ATIF trajectories."""

from pydantic import BaseModel, Field

from agent_apprenticeship.atif_models.observation_result import ObservationResult


class Observation(BaseModel):
    """Environment feedback/result after actions or system events."""

    results: list[ObservationResult] = Field(
        default=...,
        description="Array of result objects from tool calls or actions",
    )

    model_config = {"extra": "forbid"}
