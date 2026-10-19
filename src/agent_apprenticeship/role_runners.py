from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
class RoleResult(BaseModel):
    model_config=ConfigDict(extra='forbid')
    role: str; provider: str; model: str; live_call_ok: bool; structured_output_validation_ok: bool; prompt_ref: str; output_ref: str; parsed_output_ref: str; error_type: str | None = None; error_message: str | None = None; duration_seconds: float | None = None; metadata_json: dict = Field(default_factory=dict)
class RoleRunner:
    def run(self, *args, **kwargs): raise NotImplementedError
