from __future__ import annotations

from pathlib import Path
from typing import Any

from .env import redact_secrets
from .io import write_json


EXPECTED_AGENT_CONTRACT = ["agent_trace.json", "actual_outputs.json", "artifacts/"]


def build_contract_diagnostics(
    attempt_dir: Path,
    *,
    command: list[str] | str,
    working_directory: Path,
    agent_display_name: str,
    prompt: str | None = None,
) -> dict[str, Any]:
    command_value: list[str] | str
    if isinstance(command, list):
        command_value = ["<prompt>" if prompt is not None and part == prompt else redact_secrets(str(part)) for part in command]
    else:
        command_value = redact_secrets(command)
    expected = {
        "agent_trace.json": attempt_dir / "agent_trace.json",
        "actual_outputs.json": attempt_dir / "actual_outputs.json",
        "artifacts/": attempt_dir / "artifacts",
    }
    missing = [name for name, path in expected.items() if not path.exists()]
    top_level = sorted(p.name + ("/" if p.is_dir() else "") for p in attempt_dir.iterdir()) if attempt_dir.exists() else []
    likely_cause = (
        f"{agent_display_name} ran but did not write the Agent Apprenticeship output contract "
        "in the current task workspace."
        if missing
        else f"{agent_display_name} wrote the expected output contract."
    )
    if missing and any(name in {"stdout.txt", "stderr.txt", "final_message.txt"} for name in top_level):
        likely_cause += " Inspect stdout/stderr/final_message for setup, auth, or unsupported headless-mode details."
    diagnostics = {
        "command_used": command_value,
        "working_directory": str(working_directory),
        "expected_files": EXPECTED_AGENT_CONTRACT,
        "missing_expected_files": missing,
        "top_level_files_found": top_level,
        "stdout_ref": "stdout.txt",
        "stderr_ref": "stderr.txt",
        "final_message_ref": "final_message.txt",
        "likely_cause": likely_cause,
    }
    write_json(attempt_dir / "contract_diagnostics.json", diagnostics)
    return diagnostics


def diagnostics_text(diagnostics: dict[str, Any]) -> str:
    lines = [
        "Agent Apprenticeship output-contract diagnostics",
        f"Command used: {diagnostics.get('command_used')}",
        f"Working directory: {diagnostics.get('working_directory')}",
        "Expected files: " + ", ".join(diagnostics.get("expected_files") or []),
        "Missing expected files: " + (", ".join(diagnostics.get("missing_expected_files") or []) or "none"),
        "Top-level files found: " + (", ".join(diagnostics.get("top_level_files_found") or []) or "none"),
        f"stdout: {diagnostics.get('stdout_ref')}",
        f"stderr: {diagnostics.get('stderr_ref')}",
        f"final message: {diagnostics.get('final_message_ref')}",
        f"Likely cause: {diagnostics.get('likely_cause')}",
    ]
    return "\n".join(lines) + "\n"
