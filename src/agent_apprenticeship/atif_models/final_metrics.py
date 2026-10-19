# Harbor ATIF models (Apache-2.0; see LICENSE).
# Source: harbor-framework/harbor@357c5e74020fc0890d69fb3a7db610896b9ce57d
# Modified: local imports; trajectory schema version pinned to v1.8.
"""Final metrics model for ATIF trajectories."""

from typing import Any

from pydantic import BaseModel, Field


class FinalMetrics(BaseModel):
    """Aggregate statistics for the entire trajectory."""

    total_prompt_tokens: int | None = Field(
        default=None,
        description="Sum of all prompt tokens across all steps, including cached tokens",
    )
    total_completion_tokens: int | None = Field(
        default=None,
        description="Sum of all completion tokens across all steps",
    )
    total_cached_tokens: int | None = Field(
        default=None,
        description="Sum of all cached tokens across all steps",
    )
    total_cost_usd: float | None = Field(
        default=None,
        description="Total real monetary cost for the entire trajectory, including cost for subagents, if any",
    )
    total_steps: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Total number of steps. If not equivalent to the number of steps in the "
            "trajectory, must be documented in the root-level notes field."
        ),
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom aggregate metrics",
    )

    model_config = {"extra": "forbid"}
