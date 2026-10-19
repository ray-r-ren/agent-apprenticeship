from __future__ import annotations

import json
import shutil
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path

from .actual_outputs_normalizer import ActualOutputsNormalizationContext, repair_actual_outputs_file, write_actual_outputs_normalization
from .contract_diagnostics import build_contract_diagnostics, diagnostics_text
from .codex_runner import (
    AttemptResult,
    _attempt_dir,
    _copy_attempt_inputs,
    _deliverables,
    _run_with_process_group_timeout,
    ensure_attempt_outputs,
)
from .config import get_settings
from .command_discovery import resolve_command
from .env import redact_secrets
from .io import write_json
from .recipes import WORKER_AGENT_RECIPES
from .schemas import ActualOutputs, RawTaskRecord, TaskIntakeSpec
from .trace_prompt import build_worker_prompt


@dataclass
class AgentInvocation:
    argv: list[str]
    mode: str
    unsupported_reason: str | None = None


def _help_text(command: str, *args: str) -> str:
    try:
        cp = subprocess.run([command, *args, "--help"], cwd=None, text=True, capture_output=True, timeout=5)
        return f"{cp.stdout or ''}\n{cp.stderr or ''}"
    except Exception:
        return ""


def _first_json_array(text: str) -> list[object] | None:
    decoder = json.JSONDecoder()
    start = text.find("[")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("[", start + 1)
            continue
        return value if isinstance(value, list) else None
    return None


def _openclaw_default_agent_id(command: str) -> str | None:
    try:
        cp = subprocess.run([command, "agents", "list", "--json"], cwd=None, text=True, capture_output=True, timeout=5)
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    agents = _first_json_array(f"{cp.stdout or ''}\n{cp.stderr or ''}")
    if not agents:
        return None
    dict_agents = [a for a in agents if isinstance(a, dict) and a.get("id")]
    for agent in dict_agents:
        if agent.get("isDefault") is True:
            return str(agent["id"])
    return str(dict_agents[0]["id"]) if dict_agents else None


def _has_any(text: str, *needles: str) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def _has_subcommand(help_text: str, command_name: str) -> bool:
    pattern = rf"(?m)^\s{{1,}}{re.escape(command_name)}(?:\s|$)"
    return (
        bool(re.search(pattern, help_text))
        or bool(re.search(rf"(?m)^\s*{re.escape(command_name)}\s+\*", help_text))
        or bool(re.search(rf"(?m)^\s*\S+\s+{re.escape(command_name)}(?:\s|$)", help_text))
    )


def _with_workspace(argv: list[str], help_text: str, workspace: Path) -> list[str]:
    if "--workspace" in help_text:
        return [*argv, "--workspace", str(workspace)]
    if "--cwd" in help_text:
        return [*argv, "--cwd", str(workspace)]
    if "--cd" in help_text:
        return [*argv, "--cd", str(workspace)]
    if "--dir" in help_text:
        return [*argv, "--dir", str(workspace)]
    return argv


def build_agent_invocation(agent_id: str, command: str, workspace: Path, prompt_file: Path, prompt: str) -> AgentInvocation:
    help_text = _help_text(command)
    if agent_id == "cursor":
        base = _with_workspace([command], help_text, workspace)
        if "--trust" in help_text:
            base.append("--trust")
        if "--force" in help_text:
            base.append("--force")
        if "--prompt-file" in help_text:
            return AgentInvocation([*base, "--prompt-file", str(prompt_file)], "cursor-agent --prompt-file")
        if "--prompt" in help_text:
            return AgentInvocation([*base, "--prompt", prompt], "cursor-agent --prompt")
        if " -p" in help_text or "-p," in help_text:
            return AgentInvocation([*base, "-p", prompt], "cursor-agent -p")
        if _has_subcommand(help_text, "run"):
            run_help = _help_text(command, "run")
            if "--prompt-file" in run_help:
                return AgentInvocation([*_with_workspace([command, "run"], run_help, workspace), "--prompt-file", str(prompt_file)], "cursor-agent run --prompt-file")
            return AgentInvocation([*_with_workspace([command, "run"], run_help, workspace), prompt], "cursor-agent run")
        return AgentInvocation([], "cursor-agent", "Cursor headless mode unavailable: cursor-agent help did not expose --prompt-file, --prompt, -p, or run.")

    if agent_id == "claude-code":
        base = [command]
        if "--permission-mode" in help_text:
            base.extend(["--permission-mode", "bypassPermissions"])
        if "--output-format" in help_text:
            base.extend(["--output-format", "text"])
        if "--max-budget-usd" in help_text:
            base.extend(["--max-budget-usd", "1"])
        if " -p" in help_text or "-p," in help_text:
            return AgentInvocation([*base, "-p", prompt], "claude -p")
        if "--print" in help_text:
            return AgentInvocation([*base, "--print", prompt], "claude --print")
        return AgentInvocation([], "claude", "Claude Code headless mode unavailable: claude --help did not expose -p or --print.")

    if agent_id == "opencode":
        run_help = _help_text(command, "run") if _has_subcommand(help_text, "run") else ""
        if run_help:
            if "--prompt-file" in run_help:
                return AgentInvocation([*_with_workspace([command, "run"], run_help, workspace), "--prompt-file", str(prompt_file)], "opencode run --prompt-file")
            if "--prompt" in run_help:
                return AgentInvocation([*_with_workspace([command, "run"], run_help, workspace), "--prompt", prompt], "opencode run --prompt")
            return AgentInvocation([*_with_workspace([command, "run"], run_help, workspace), prompt], "opencode run")
        return AgentInvocation([], "opencode", "OpenCode headless mode unavailable: opencode --help did not expose a run command.")

    if agent_id == "openclaw":
        if _has_subcommand(help_text, "agent"):
            agent_help = _help_text(command, "agent")
            if "--message" in agent_help or "-m," in agent_help:
                settings = get_settings()
                agent_ref = settings.worker_agent_model or _openclaw_default_agent_id(command)
                if not agent_ref and "--agent" in agent_help:
                    return AgentInvocation(
                        [],
                        "openclaw agent",
                        "OpenClaw setup required: no configured OpenClaw agent was found. Run `openclaw setup` or `openclaw agents add`, then rerun the smoke.",
                    )
                argv = [command, "agent"]
                if "--local" in agent_help:
                    argv.append("--local")
                if "--json" in agent_help:
                    argv.append("--json")
                if "--agent" in agent_help and agent_ref:
                    argv.extend(["--agent", agent_ref])
                if "--timeout" in agent_help:
                    argv.extend(["--timeout", str(settings.task_timeout_seconds)])
                argv.extend(["--message", prompt])
                return AgentInvocation(argv, "openclaw agent --local --message")
            return AgentInvocation([], "openclaw agent", "OpenClaw headless execution is unavailable in this installed version: openclaw agent --help did not expose --message.")
        for sub in ("run", "exec", "session"):
            if _has_subcommand(help_text, sub):
                sub_help = _help_text(command, sub)
                if "--prompt-file" in sub_help:
                    return AgentInvocation([*_with_workspace([command, sub], sub_help, workspace), "--prompt-file", str(prompt_file)], f"openclaw {sub} --prompt-file")
                if "--prompt" in sub_help:
                    return AgentInvocation([*_with_workspace([command, sub], sub_help, workspace), "--prompt", prompt], f"openclaw {sub} --prompt")
                return AgentInvocation([*_with_workspace([command, sub], sub_help, workspace), prompt], f"openclaw {sub}")
        return AgentInvocation([], "openclaw", "OpenClaw headless execution is unavailable in this installed version: openclaw --help did not expose agent, run, exec, or session.")

    if agent_id == "hermes-agent":
        for sub in ("run", "chat"):
            if _has_subcommand(help_text, sub):
                sub_help = _help_text(command, sub)
                base = [command, sub]
                if "--quiet" in sub_help:
                    base.append("--quiet")
                if "--yolo" in sub_help:
                    base.append("--yolo")
                if "--max-turns" in sub_help:
                    base.extend(["--max-turns", "12"])
                if "--prompt-file" in sub_help:
                    return AgentInvocation([*_with_workspace(base, sub_help, workspace), "--prompt-file", str(prompt_file)], f"hermes {sub} --prompt-file")
                if "--prompt" in sub_help:
                    return AgentInvocation([*_with_workspace(base, sub_help, workspace), "--prompt", prompt], f"hermes {sub} --prompt")
                if "--query" in sub_help:
                    return AgentInvocation([*_with_workspace(base, sub_help, workspace), "--query", prompt], f"hermes {sub} --query")
                if " -q" in sub_help or "-q," in sub_help:
                    return AgentInvocation([*_with_workspace(base, sub_help, workspace), "-q", prompt], f"hermes {sub} -q")
                return AgentInvocation([*_with_workspace(base, sub_help, workspace), prompt], f"hermes {sub}")
        return AgentInvocation([], "hermes", "Hermes Agent headless mode unavailable: hermes --help did not expose run or chat.")

    return AgentInvocation([], agent_id, f"Unsupported Apprentice Agent adapter: {agent_id}")


def classify_agent_failure(agent_id: str, display_name: str, error: object | None, stdout: str = "", stderr: str = "", returncode: int | None = None) -> str | None:
    text = f"{error or ''}\n{stdout or ''}\n{stderr or ''}".lower()
    command = WORKER_AGENT_RECIPES.get(agent_id).command_name if agent_id in WORKER_AGENT_RECIPES else agent_id
    if error is None and returncode == 0:
        return None
    if isinstance(error, FileNotFoundError) or f"no such file or directory: '{command}'" in text or f"no such file or directory: {command}" in text:
        return f"Apprentice Agent command not found: {command}"
    if "headless mode unavailable" in text or "headless execution is unavailable" in text:
        return str(error or stderr).strip()
    if isinstance(error, subprocess.TimeoutExpired) or "timed out" in text or "timeout expired" in text or "command timed out" in text:
        return f"Apprentice Agent attempt timed out while running {display_name}."
    if _has_any(text, "eperm", "permission denied", "operation not permitted", "read-only file system"):
        return f"Apprentice Agent permission error while running {display_name}: the external CLI could not access a required file or directory."
    if _has_any(
        text,
        "not authenticated",
        "not logged in",
        "login required",
        "please login",
        "please log in",
        "authentication failed",
        "unauthorized",
        "invalid api key",
        "missing api key",
        "api key missing",
        "api key is missing",
        "api key not configured",
        "api key is not configured",
        "provider api key is missing",
        "google generative ai api key is missing",
    ):
        return f"{display_name} setup required: authentication, API key, or account setup is required."
    if _has_any(text, "setup required", "onboarding", "configure first", "model not configured", "provider not configured"):
        return f"{display_name} setup required: complete the provider/model setup before running."
    if agent_id == "openclaw" and _has_any(text, "unknown agent id", "pass --to", "session-id", "choose a session", "no configured openclaw agent"):
        return "OpenClaw setup required: configure an OpenClaw agent with `openclaw setup` or `openclaw agents add`, then rerun the smoke."
    if _has_any(text, "quota", "rate limit", "billing", "insufficient quota", "insufficient credits", "out of credits", "credit limit", "usage limit"):
        return f"{display_name} provider quota or credit limit reached."
    if returncode not in (None, 0):
        return f"Apprentice Agent exited before producing required outputs (exit code {returncode})."
    if error:
        return f"Apprentice Agent operational error: {redact_secrets(str(error))}"
    return None


def run_external_agent_attempt(package_root: Path, raw: RawTaskRecord, spec: TaskIntakeSpec, attempt_kind: str = "baseline", timeout: int | None = None) -> AttemptResult:
    settings = get_settings()
    agent_id = settings.worker_agent
    recipe = WORKER_AGENT_RECIPES[agent_id]
    display = recipe.display_name
    command = settings.worker_agent_command or recipe.command_name
    d = _attempt_dir(package_root, attempt_kind)
    input_files = _copy_attempt_inputs(package_root, d)
    deliverables = _deliverables(raw, spec)
    rubric_md = (package_root / "rubric" / "worker_visible_rubric.md").read_text() if (package_root / "rubric" / "worker_visible_rubric.md").exists() else ""
    prompt = build_worker_prompt(spec.normalized_instruction, rubric_md, attempt_kind, input_files, deliverables, settings.sensitive_info_masking, workspace_path=str(d))
    prompt_file = d / "prompt.md"
    prompt_file.write_text(prompt)

    run_error: object | None = None
    returncode: int | None = None
    stdout = ""
    stderr = ""
    invocation = AgentInvocation([], agent_id)
    resolved_command = resolve_command(command)
    if not resolved_command:
        run_error = FileNotFoundError(command)
    else:
        invocation = build_agent_invocation(agent_id, resolved_command, d, prompt_file, prompt)
        if invocation.unsupported_reason:
            run_error = RuntimeError(invocation.unsupported_reason)
        else:
            try:
                cp = _run_with_process_group_timeout(invocation.argv, cwd=d, timeout=timeout or settings.task_timeout_seconds)
                returncode = cp.returncode
                stdout = cp.stdout or ""
                stderr = cp.stderr or ""
                if cp.returncode != 0:
                    run_error = RuntimeError(classify_agent_failure(agent_id, display, None, stdout, stderr, cp.returncode) or f"{display} exited with code {cp.returncode}.")
            except Exception as exc:
                run_error = exc
    (d / "stdout.txt").write_text(redact_secrets(stdout))
    (d / "stderr.txt").write_text(redact_secrets(stderr if stderr else str(run_error or "")))
    (d / "final_message.txt").write_text(redact_secrets((stdout or stderr or str(run_error or ""))[-4000:]))
    contract_missing_before_repair = not (d / "agent_trace.json").exists() or not (d / "actual_outputs.json").exists()
    contract_diagnostics = None
    if contract_missing_before_repair:
        command_for_diagnostics = invocation.argv or [command]
        contract_diagnostics = build_contract_diagnostics(
            d,
            command=command_for_diagnostics,
            working_directory=d,
            agent_display_name=display,
            prompt=prompt,
        )
        with (d / "final_message.txt").open("a") as f:
            f.write("\n\n" + diagnostics_text(contract_diagnostics))

    trace, actual, trace_valid = ensure_attempt_outputs(package_root, spec, attempt_kind, prompt, agent_id, run_error if isinstance(run_error, Exception) else None)
    if actual.metadata_json is None:
        actual.metadata_json = {}
    actual.metadata_json["apprentice_agent"] = agent_id
    actual.metadata_json["apprentice_agent_display_name"] = display
    actual.metadata_json["apprentice_agent_invocation"] = [part if part != prompt else "<prompt>" for part in invocation.argv]
    actual.metadata_json["apprentice_agent_invocation_mode"] = invocation.mode
    if contract_diagnostics:
        actual.metadata_json["apprentice_agent_contract_diagnostics"] = contract_diagnostics
    classified_error = classify_agent_failure(agent_id, display, run_error, stdout, stderr, returncode)
    output_contract_error = (
        f"Apprentice Agent output-contract failure: {display} did not produce required "
        "agent_trace.json and actual_outputs.json."
    )
    if not trace_valid:
        operational_prefixes = (
            "Apprentice Agent command not found",
            f"{display} setup required",
            f"{display} provider quota",
            "Apprentice Agent timed out",
            "Apprentice Agent attempt timed out",
            "Apprentice Agent permission error",
        )
        if classified_error and (classified_error.startswith(operational_prefixes) or "headless mode unavailable" in classified_error or "headless execution is unavailable" in classified_error):
            op_error = classified_error
        else:
            op_error = output_contract_error
    else:
        op_error = classified_error
    if op_error and returncode not in (None, 0) and trace_valid and actual.status == "success":
        op_error = f"Apprentice Agent exited nonzero after producing required outputs (exit code {returncode})."
    if op_error:
        actual.metadata_json["apprentice_agent_operational_error"] = op_error
        actual.error_message = op_error
    write_json(d / "actual_outputs.json", actual)

    trace.metadata_json["apprentice_agent"] = agent_id
    trace.metadata_json["apprentice_agent_invocation"] = [part if part != prompt else "<prompt>" for part in invocation.argv]
    trace.metadata_json["trace_valid"] = trace_valid
    if op_error:
        trace.metadata_json["apprentice_agent_operational_error"] = op_error
    if contract_diagnostics:
        trace.metadata_json["apprentice_agent_contract_diagnostics"] = contract_diagnostics
    write_json(d / "agent_trace.json", trace)
    actual_ctx = ActualOutputsNormalizationContext(task_id=spec.task_id, attempt_id=f"{spec.task_id}_{attempt_kind}", attempt_kind=attempt_kind, package_root=package_root, required_artifacts=deliverables)
    actual_result = repair_actual_outputs_file(d / "actual_outputs.json", actual_ctx)
    write_actual_outputs_normalization(d, actual_result)
    if actual_result.actual_outputs is not None:
        actual = ActualOutputs.model_validate(actual_result.actual_outputs)
    return AttemptResult(attempt_dir=str(d), trace_valid=trace_valid, trace=trace, actual_outputs=actual, apprentice_agent=agent_id)
