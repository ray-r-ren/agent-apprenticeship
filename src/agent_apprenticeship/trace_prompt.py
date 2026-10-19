from __future__ import annotations

TRACE_SCHEMA_GUIDANCE = """Write ./agent_trace.json as a complete Harbor ATIF v1.8 trajectory:
{
  "schema_version": "ATIF-v1.8",
  "agent": {"name": "your-agent", "version": "your-agent-version"},
  "steps": [
    {"step_id": 1, "source": "user", "message": "Task instruction"},
    {"step_id": 2, "source": "agent", "message": "Read task input",
     "tool_calls": [{"tool_call_id": "call_2", "function_name": "read_file", "arguments": {"path": "input/task.md"}}],
     "observation": {"results": [{"source_call_id": "call_2", "content": "Observed result"}]}}
  ]
}
Use sequential integer step_id values starting at 1; source is system, user, or agent.
Record each meaningful interaction faithfully. Tool results belong in observation.results,
with source_call_id matching a tool_call_id on the same step. Preserve parallel calls.
Record available timestamps, model_name, metrics, final_metrics and subagent_trajectories.
Embedded subagents need unique trajectory_id values. References use trajectory_id or
trajectory_path. Image/audio message parts use type and source {media_type, path};
audio may include duration_sec. Never embed media bytes or invent results or metrics.
Put project-specific task IDs, attempt IDs, artifacts, causal links, surface fields,
and outcome evidence in extra.agent_apprenticeship at the root or step level.
Do not add non-ATIF top-level fields. Do not include hidden chain-of-thought or secrets.
Read inputs under ./input/ and write deliverables under ./artifacts/.
Also write ./actual_outputs.json with task_id, attempt_id, attempt_kind, status,
output_summary, primary_output_ref, deliverable_refs, artifact_refs and files_created.
Use package-relative artifact references. Evaluation is performed separately.
"""

MASKING_GUIDANCE = """Sensitive info masking is standard for this run. Do not expose API keys, private tokens, credentials, or secrets in traces, logs, or artifacts. Preserve useful operational context while masking super-sensitive values."""


def build_worker_prompt(
    task_instruction: str,
    rubric_md: str = "",
    attempt_kind: str = "baseline",
    input_files: list[str] | None = None,
    deliverables: list[str] | None = None,
    sensitive_info_masking: str = "standard",
    workspace_path: str | None = None,
) -> str:
    input_files = input_files or []
    deliverables = deliverables or []
    input_section = "\n".join(f"- ./input/{name}" for name in input_files) or "- No explicit input files declared."
    deliverable_section = "\n".join(f"- ./artifacts/{name}" for name in deliverables) or "- Write required task deliverables under ./artifacts/."
    workspace_section = f"\nCurrent task workspace: {workspace_path}\n" if workspace_path else ""
    masking_section = f"\n\n## Sensitive info masking\n{MASKING_GUIDANCE}" if sensitive_info_masking == "standard" else ""
    return f"# Agent Apprenticeship Workflow Task\n\nAttempt: {attempt_kind}\n{workspace_section}\n{TRACE_SCHEMA_GUIDANCE}{masking_section}\n\n## Input files\n{input_section}\n\n## Required output contract\nYou must write these files in the current task workspace before finishing:\n- ./agent_trace.json\n- ./actual_outputs.json\n- ./artifacts/\n{deliverable_section}\n\nBefore finishing, verify ./agent_trace.json, ./actual_outputs.json, and ./artifacts/ exist. Do not only answer in chat/stdout. Do not say the task is complete unless those files exist.\n\n## Task\n{task_instruction}\n\n## Worker-visible evaluation rubric\n{rubric_md}\n\nFinal instruction: write the required files in this workspace, verify they exist, then stop.\n"
