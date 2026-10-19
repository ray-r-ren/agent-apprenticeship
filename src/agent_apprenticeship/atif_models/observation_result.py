# Harbor ATIF models (Apache-2.0; see LICENSE).
# Source: harbor-framework/harbor@357c5e74020fc0890d69fb3a7db610896b9ce57d
# Modified: local imports; trajectory schema version pinned to v1.8.
"""Observation result model for ATIF trajectories."""

from typing import Any

from pydantic import BaseModel, Field

from agent_apprenticeship.atif_models.content import ContentPart
from agent_apprenticeship.atif_models.subagent_trajectory_ref import SubagentTrajectoryRef


class ObservationResult(BaseModel):
    """A single result within an observation."""

    source_call_id: str | None = Field(
        default=None,
        description=(
            "The `tool_call_id` from the _tool_calls_ array in _StepObject_ that this "
            "result corresponds to. If null or omitted, the result comes from an "
            "action that doesn't use the standard tool calling format (e.g., agent "
            "actions without tool calls or system-initiated operations)."
        ),
    )
    content: str | list[ContentPart] | None = Field(
        default=None,
        description=(
            "The output or result from the tool execution. String for text-only "
            "content, or array of ContentPart for multimodal content (added in ATIF-v1.6)."
        ),
    )
    subagent_trajectory_ref: list[SubagentTrajectoryRef] | None = Field(
        default=None,
        description="Array of references to delegated subagent trajectories",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Custom observation-result-level metadata (e.g., confidence score, "
            "retrieval score, source document ID). Added in ATIF-v1.7."
        ),
    )

    model_config = {"extra": "forbid"}
