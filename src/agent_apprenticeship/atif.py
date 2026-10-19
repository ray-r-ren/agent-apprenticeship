"""ATIF v1.8 interchange and lossless projections for workflow evaluation.

Only ATIF documents cross persistence boundaries. Project-specific evidence lives
in ``extra.agent_apprenticeship``; Harbor fields are preserved independently of
our task/evaluation projection, including parallel calls and multimodal content.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from .atif_models import Trajectory

VERSION = "ATIF-v1.8"
NAMESPACE = "agent_apprenticeship"


def validate_trace(data: Any) -> dict:
    if not isinstance(data, dict) or data.get("schema_version") != VERSION:
        raise ValueError("Trace must declare schema_version ATIF-v1.8")
    return Trajectory.model_validate(data).to_json_dict()


def load_trace(path: Path) -> dict:
    return validate_trace(json.loads(path.read_text(encoding="utf-8")))


def dump_trace(path: Path, trace: Any) -> None:
    if isinstance(trace, Trajectory):
        trace = trace.to_json_dict()
    data = validate_trace(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def to_workflow(data: dict) -> dict:
    """Project validated ATIF into the internal evaluator's task/step view."""
    data = validate_trace(data)
    meta = deepcopy((data.get("extra") or {}).get(NAMESPACE) or {})
    # Filter extension fields for the strongly typed task projection.
    from .schemas import WorkflowTrace, WorkflowTraceStep
    meta = {k: v for k, v in meta.items() if k in WorkflowTrace.model_fields}
    identity = data.get("trajectory_id") or data.get("session_id") or "external"
    meta.setdefault("trace_id", identity)
    meta.setdefault("run_id", data.get("session_id"))
    meta.setdefault("task_id", identity)
    meta.setdefault("attempt_id", identity)
    meta.setdefault("attempt_kind", "other")
    meta.setdefault("trace_mode", "live")
    meta.setdefault("task", next((_text(s["message"]) for s in data["steps"] if s["source"] == "user"), ""))
    meta.setdefault("agent_tools", sorted({c["function_name"] for s in data["steps"] for c in s.get("tool_calls") or []}))
    meta.setdefault("termination_reason", "other")
    meta["schema_version"] = VERSION
    meta.setdefault("metadata_json", {})["_atif_original"] = deepcopy(data)
    steps = []
    for s in data["steps"]:
        ext = deepcopy((s.get("extra") or {}).get(NAMESPACE) or {})
        ext = {k: v for k, v in ext.items() if k in WorkflowTraceStep.model_fields}
        ext.update(step=s["step_id"])
        ext.setdefault("turn", s["step_id"])
        ext.setdefault("actor", "user" if s["source"] == "user" else "agent:worker")
        ext.setdefault("action", "user_message" if s["source"] == "user" else "agent_step")
        ext.setdefault("input", _text(s["message"]))
        if s["source"] != "user":
            ext.setdefault("operation", "other")
            calls = s.get("tool_calls") or []
            ext.setdefault("tool", calls[0]["function_name"] if len(calls) == 1 else None)
            results = (s.get("observation") or {}).get("results") or []
            ext.setdefault("output", "\n".join(_text(r.get("content")) for r in results) or _text(s["message"]))
            ext.setdefault("reasoning", s.get("reasoning_content"))
        steps.append(ext)
    meta["steps"] = steps
    return meta


def from_workflow(trace: dict) -> dict:
    """Serialize a task projection as ATIF, retaining original Harbor content."""
    trace = deepcopy(trace)
    original = (trace.get("metadata_json") or {}).pop("_atif_original", None)
    out = deepcopy(original) if original else {
        "schema_version": VERSION,
        "agent": {"name": "agent-apprenticeship", "version": "0.3.0"},
    }
    out["session_id"] = trace.get("run_id") or out.get("session_id") or trace.get("attempt_id") or trace["trace_id"]
    out["trajectory_id"] = trace["trace_id"]
    steps = []
    for i, s in enumerate(trace.pop("steps"), 1):
        prior = original["steps"][i-1] if original and i <= len(original["steps"]) else None
        step = deepcopy(prior) if prior else {
            "step_id": i,
            "source": "user" if s.get("actor") == "user" else "system" if s.get("actor") == "system" else "agent",
            "message": _text(s.get("input") if s.get("actor") == "user" else s.get("output") or s.get("input") or s.get("action")),
        }
        step["step_id"] = i
        if not prior:
            if s.get("started_at"):
                step["timestamp"] = s["started_at"]
            if step["source"] == "agent":
                if s.get("reasoning"):
                    step["reasoning_content"] = s["reasoning"]
                if s.get("tool"):
                    call_id = f"call_{i}"
                    step["tool_calls"] = [{"tool_call_id": call_id, "function_name": s["tool"], "arguments": {"input": s.get("input") or ""}}]
                    if s.get("output") is not None:
                        step["observation"] = {"results": [{"source_call_id": call_id, "content": s["output"]}]}
        step.setdefault("extra", {}).setdefault(NAMESPACE, {}).update({k: v for k, v in s.items() if v is not None and v != [] and v != {}})
        steps.append(step)
    out["steps"] = steps
    trace.pop("schema_version", None)
    out.setdefault("extra", {}).setdefault(NAMESPACE, {}).update({k: v for k, v in trace.items() if v is not None and v != [] and v != {}})
    return validate_trace(out)


def encode_traces(data: Any) -> Any:
    """Encode trace objects even when nested in training/evaluation envelopes."""
    if isinstance(data, list):
        return [encode_traces(v) for v in data]
    if not isinstance(data, dict):
        return data
    if isinstance(data.get("steps"), list) and "agent" in data:
        return validate_trace(data)
    if isinstance(data.get("steps"), list) and "trace_id" in data:
        return from_workflow(data)
    return {k: encode_traces(v) for k, v in data.items()}


def decode_traces(data: Any) -> Any:
    if isinstance(data, list):
        return [decode_traces(v) for v in data]
    if not isinstance(data, dict):
        return data
    if isinstance(data.get("steps"), list) and "agent" in data:
        return to_workflow(data)
    return {k: decode_traces(v) for k, v in data.items()}


def event_trace(event: dict, name: str) -> dict:
    """Represent a compiler/session event as a complete one-step trajectory."""
    return validate_trace({
        "schema_version": VERSION,
        "session_id": event.get("session_id") or event.get("run_id"),
        "agent": {"name": name, "version": "0.3.0"},
        "steps": [{"step_id": 1, "source": "system", "message": _text(event.get("message") or event.get("instruction") or event.get("event_type")), "extra": {NAMESPACE: event}}],
        "extra": {NAMESPACE: {"event": event}},
    })
