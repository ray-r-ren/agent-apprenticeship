from __future__ import annotations
import os
import signal
import shutil, subprocess
from pathlib import Path
import re
from .schemas import WorkflowTrace, WorkflowTraceStep, ActualOutputs, RawTaskRecord, TaskIntakeSpec
from .trace_prompt import build_worker_prompt
from .io import write_json
from .config import get_settings
from .env import redact_secrets
from .contract_diagnostics import build_contract_diagnostics, diagnostics_text
from .trace_normalizer import TraceNormalizationContext, repair_agent_trace_file, TraceNormalizationReport
from .actual_outputs_normalizer import ActualOutputsNormalizationContext, repair_actual_outputs_file, write_actual_outputs_normalization
from .public_sanitizer import classify_provider_failure, public_error_summary
from .progress import utc_now

class AttemptResult(dict): pass

CODEX_TRUST_RETRY_MESSAGE = "Codex refused to run because the workspace is not a trusted Git directory. Retrying with --skip-git-repo-check if supported."


def _safe_agent_config(settings, runner_kind: str, attempt_kind: str) -> dict:
    return {
        "apprentice_agent_id": settings.worker_agent,
        "apprentice_agent_name": settings.custom_worker_display_name if settings.worker_agent == "custom" else settings.worker_agent,
        "apprenticeship_mode": getattr(settings, "apprenticeship_mode", None),
        "mentor_mode": settings.mentor_mode,
        "mentor_model_provider": settings.model_provider,
        "max_improvement_loops": settings.max_improvement_loops,
        "attempt_kind": attempt_kind,
        "runner_kind": runner_kind,
        "sensitive_info_masking": settings.sensitive_info_masking,
    }


def _enrich_runtime_trace(
    trace: WorkflowTrace,
    actual: ActualOutputs,
    package_root: Path,
    attempt_kind: str,
    runner_kind: str,
    started_at: str | None,
    ended_at: str | None,
) -> WorkflowTrace:
    settings = get_settings()
    trace.run_id = package_root.parent.parent.name if package_root.parent.parent.name else None
    trace.package_id = package_root.name
    trace.started_at = trace.started_at or started_at
    trace.ended_at = trace.ended_at or ended_at
    trace.agent_config = trace.agent_config or _safe_agent_config(settings, runner_kind, attempt_kind)
    trace.actual_outputs_ref = f"attempts/{attempt_kind}/actual_outputs.json"
    trace.input_artifact_refs = list(dict.fromkeys(actual.input_artifact_refs or trace.input_artifact_refs or []))
    trace.artifact_refs = list(dict.fromkeys((actual.artifact_refs or actual.files_created or []) + (trace.artifact_refs or [])))
    trace.deliverable_refs = list(dict.fromkeys((actual.deliverable_refs or []) + (trace.deliverable_refs or [])))
    trace.output_summary = trace.output_summary or actual.output_summary
    trace.final_output_summary = trace.final_output_summary or actual.output_summary
    trace.environment_domain = trace.environment_domain or (
        "mixed"
        if len({s.environment_domain for s in trace.steps if s.environment_domain and s.environment_domain != "unknown"}) > 1
        else next(iter({s.environment_domain for s in trace.steps if s.environment_domain and s.environment_domain != "unknown"}), "unknown")
    )
    trace.metadata_json.setdefault("runtime_metadata_populated", True)
    return trace


def _run_with_process_group_timeout(command, *, cwd: Path, timeout: int | None, shell: bool = False) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            process.kill()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from exc

def _attempt_dir(package_root: Path, attempt_kind: str) -> Path:
    p=package_root/'attempts'/attempt_kind
    (p/'artifacts').mkdir(parents=True, exist_ok=True)
    (p/'input').mkdir(parents=True, exist_ok=True)
    return p


def _recover_misplaced_contract_files(d: Path) -> dict[str, str]:
    """Recover contract files agents sometimes place under ./artifacts/."""
    recovered: dict[str, str] = {}
    for name in ("agent_trace.json", "actual_outputs.json"):
        dst = d / name
        src = d / "artifacts" / name
        if dst.exists() or not src.exists() or not src.is_file():
            continue
        shutil.copy2(src, dst)
        recovered[name] = f"artifacts/{name}"
    return recovered

def _copy_attempt_inputs(package_root: Path, attempt_dir: Path) -> list[str]:
    src=package_root/'input'; dst=attempt_dir/'input'
    dst.mkdir(parents=True, exist_ok=True)
    copied=[]
    if src.exists():
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst/f.name); copied.append(f.name)
            elif f.is_dir():
                shutil.copytree(f, dst/f.name, dirs_exist_ok=True); copied.append(f.name)
    return copied

def parse_expected_deliverables(text: str | None) -> list[str]:
    if not text:
        return []
    parts=re.split(r'\s*(?:\+|,|;|\band\b|\n)\s*', str(text))
    items=[]
    for part in parts:
        part=part.strip().strip('`*•- ')
        if not part:
            continue
        match=re.search(r'[A-Za-z0-9_.-]+\.(?:csv|xlsx|xls|json|jsonl|md|txt|pdf|html|xml|py|zip)', part, re.I)
        if match:
            items.append(match.group(0))
    return list(dict.fromkeys(items))

def _deliverables(raw: RawTaskRecord, spec: TaskIntakeSpec) -> list[str]:
    expected=raw.expected_deliverable or raw.raw_payload.get('expected_deliverable')
    parsed=parse_expected_deliverables(expected)
    if parsed:
        return parsed
    reqs=raw.raw_payload.get('output_requirements') or spec.output_requirements
    file_like=[]
    for item in reqs or []:
        item=str(item)
        if re.search(r'\.(?:csv|xlsx|xls|json|jsonl|md|txt|pdf|html|xml|py|zip)\b', item, re.I):
            file_like.extend(parse_expected_deliverables(item))
    return file_like or ['output.txt']

def _fallback_actual(spec: TaskIntakeSpec, attempt_kind: str, error_type: str, error_message: str) -> ActualOutputs:
    cls=classify_provider_failure(error_message)
    et=cls.get('error_type') or error_type
    msg=public_error_summary(error_message) if cls else redact_secrets(error_message)
    return ActualOutputs(task_id=spec.task_id, attempt_id=f'{spec.task_id}_{attempt_kind}', attempt_kind=attempt_kind, status='failed', output_summary='Attempt failed or did not produce a valid agent trace; raw logs were preserved.', primary_output_ref=None, deliverable_refs=[], final_message_ref=f'attempts/{attempt_kind}/final_message.txt', artifact_refs=[f'attempts/{attempt_kind}/prompt.md', f'attempts/{attempt_kind}/stdout.txt', f'attempts/{attempt_kind}/stderr.txt', f'attempts/{attempt_kind}/final_message.txt'], files_created=[], files_modified=[], files_deleted=[], stdout_ref=f'attempts/{attempt_kind}/stdout.txt', stderr_ref=f'attempts/{attempt_kind}/stderr.txt', raw_log_refs=[f'attempts/{attempt_kind}/stdout.txt',f'attempts/{attempt_kind}/stderr.txt',f'attempts/{attempt_kind}/final_message.txt'], error_type=et, error_message=msg, metadata_json={'trace_valid':False,'fallback_actual_outputs_created':True, **cls})


def _apprentice_operational_error(error: object | None, stdout: str = "", stderr: str = "", returncode: int | None = None) -> str | None:
    text = f"{error or ''}\n{stdout or ''}\n{stderr or ''}".lower()
    if error is None and returncode == 0:
        return None
    if isinstance(error, FileNotFoundError) or "no such file or directory: 'codex'" in text or "no such file or directory: codex" in text:
        return "Apprentice Agent command not found: codex"
    if isinstance(error, subprocess.TimeoutExpired) or "timed out" in text or "timeout expired" in text or "command timed out" in text:
        return "Apprentice Agent timed out while running Codex."
    if _is_codex_workspace_trust_error(stdout, stderr, error):
        return "Codex workspace trust error: Codex refused to run because the workspace is not a trusted Git directory. Use a Codex CLI with --skip-git-repo-check support or run from a trusted Git repository."
    if _is_codex_local_state_permission_error(stdout, stderr, error):
        return "Codex local state permission error: Codex could not initialize its local state or app-server client. Check that CODEX_HOME is writable and Codex can access its state directory."
    auth_markers = [
        "not authenticated",
        "not logged in",
        "login required",
        "please login",
        "please log in",
        "authentication failed",
        "auth failed",
        "unauthorized",
        "http 401",
        "status 401",
        "status code 401",
        "401 unauthorized",
        "missing api key",
        "api key not configured",
        "invalid api key",
        "credential error",
        "credentials not configured",
    ]
    if any(token in text for token in auth_markers):
        return "Apprentice Agent setup error: Codex is not authenticated or configured."
    quota_markers = [
        "quota",
        "rate limit",
        "billing",
        "insufficient quota",
        "insufficient credits",
        "out of credits",
        "credit limit",
        "usage limit",
    ]
    if any(token in text for token in quota_markers):
        return "Apprentice Agent provider quota or usage limit reached."
    if returncode not in (None, 0):
        return f"Apprentice Agent exited before producing required outputs (exit code {returncode})."
    if error:
        return f"Apprentice Agent operational error: {redact_secrets(str(error))}"
    return None


def _is_codex_workspace_trust_error(stdout: str = "", stderr: str = "", error: object | None = None) -> bool:
    text = f"{error or ''}\n{stdout or ''}\n{stderr or ''}".lower()
    return (
        "not inside a trusted directory" in text
        or "--skip-git-repo-check was not specified" in text
        or "trusted git directory" in text
    )


def _is_codex_local_state_permission_error(stdout: str = "", stderr: str = "", error: object | None = None) -> bool:
    text = f"{error or ''}\n{stdout or ''}\n{stderr or ''}".lower()
    return (
        ("codex_state" in text and "readonly database" in text)
        or ("failed to open state db" in text and "readonly database" in text)
        or ("failed to initialize in-process app-server client" in text and "operation not permitted" in text)
        or ("failed to initialize state runtime" in text and "readonly database" in text)
    )


def codex_exec_help(command: str = "codex") -> str:
    try:
        cp = subprocess.run([command, "exec", "--help"], cwd=None, text=True, capture_output=True, timeout=5)
        return (getattr(cp, "stdout", "") or "") + "\n" + (getattr(cp, "stderr", "") or "")
    except Exception:
        try:
            cp = subprocess.run([command, "--help"], cwd=None, text=True, capture_output=True, timeout=5)
            return (getattr(cp, "stdout", "") or "") + "\n" + (getattr(cp, "stderr", "") or "")
        except Exception:
            return ""


def _codex_exec_supports(flag: str, command: str = "codex") -> bool:
    return flag in codex_exec_help(command)

def _fallback_trace(spec: TaskIntakeSpec, attempt_kind: str, prompt: str, actual: ActualOutputs, error_type: str, error_message: str, codex_sandbox: str) -> WorkflowTrace:
    cls=classify_provider_failure(error_message)
    attempt_id=f'{spec.task_id}_{attempt_kind}'
    actor=f'agent:{"reviser" if attempt_kind=="revised" else "worker"}'
    safe=(public_error_summary(error_message) if cls else redact_secrets(error_message)[-3000:])
    error_type=cls.get('error_type') or error_type
    steps=[
        WorkflowTraceStep(step=1, turn=1, actor='user', action='user_message', input=spec.normalized_instruction, message_role='direct_request'),
        WorkflowTraceStep(step=2, turn=1, actor=actor, action='error', operation='other', tool='codex_cli', environment_domain='terminal', execution_mode='serial', observation='Codex attempt ended without a valid agent_trace.json or actual_outputs.json.', input='Validate required output contract for ./agent_trace.json and ./actual_outputs.json.', input_source={'source':'runtime_contract_check'}, output=safe, state_change='A minimal failure trace and failed actual_outputs.json were written by the runner.', reasoning='The package must preserve failure evidence without fabricating a successful detailed trace.', caused_by=[1], causal_type='dependency_on_tool_result', success=False, step_outcome='failed', error_type=error_type, error_message=safe, artifact_refs=[f'attempts/{attempt_kind}/prompt.md', f'attempts/{attempt_kind}/stdout.txt', f'attempts/{attempt_kind}/stderr.txt', f'attempts/{attempt_kind}/final_message.txt']),
    ]
    return WorkflowTrace(trace_id=f'trace_{attempt_id}_fallback', collection_id=None, trace_mode='live', task=spec.normalized_instruction, task_id=spec.task_id, attempt_id=attempt_id, attempt_kind=attempt_kind if attempt_kind in ['baseline','revised'] else 'other', agent_tools=['codex_cli','Bash','python','file_read','file_write'], system_prompt=prompt, system_prompt_hash=None, skills=['agent_trace_skill'], learning='When a live agent cannot write outputs, preserve logs and create a minimal failure trace.', termination_reason='agent_blocked', steps=steps, actual_outputs=actual, artifacts=[], metadata_json={'trace_valid':False,'fallback_trace_created':True,'codex_sandbox':codex_sandbox, **cls})

def _write_fallback_report(d: Path, spec: TaskIntakeSpec, attempt_kind: str, reason: str, raw_parse_error: bool=False) -> None:
    report=TraceNormalizationReport(task_id=spec.task_id, attempt_id=f'{spec.task_id}_{attempt_kind}', attempt_kind=attempt_kind, raw_trace_ref='agent_trace.raw.json' if (d/'agent_trace.raw.json').exists() else None, normalized_trace_ref=None, canonical_trace_ref='agent_trace.json', trace_schema_valid=True, trace_normalized=False, trace_lossless=True, fallback_trace=True, raw_step_count=0, normalized_step_count=2, discarded_step_count=0, raw_trace_parse_error=raw_parse_error, validation_errors=[reason], metadata_json={'fallback_reason': reason})
    write_json(d/'trace_normalization_report.json', report)


def ensure_attempt_outputs(package_root: Path, spec: TaskIntakeSpec, attempt_kind: str, prompt: str, codex_sandbox: str, validation_error: Exception | None=None) -> tuple[WorkflowTrace, ActualOutputs, bool]:
    d=_attempt_dir(package_root, attempt_kind)
    recovered_contract_files = _recover_misplaced_contract_files(d)
    actual_path=d/'actual_outputs.json'; trace_path=d/'agent_trace.json'; raw_path=d/'agent_trace.raw.json'
    err = validation_error or FileNotFoundError('agent_trace.json missing or invalid')
    required_artifacts=_deliverables(RawTaskRecord(raw_task_id=spec.task_id, source_kind='normalized_spec', raw_title=spec.normalized_title, raw_description=spec.normalized_instruction, raw_payload={'expected_deliverable': spec.expected_agent_deliverable}), spec)
    actual_ctx=ActualOutputsNormalizationContext(task_id=spec.task_id, attempt_id=f'{spec.task_id}_{attempt_kind}', attempt_kind=attempt_kind, package_root=package_root, required_artifacts=required_artifacts)
    actual_result=repair_actual_outputs_file(actual_path, actual_ctx)
    if actual_result.actual_outputs is not None:
        write_actual_outputs_normalization(d, actual_result)
        actual=ActualOutputs.model_validate(actual_result.actual_outputs)
    else:
        actual=_fallback_actual(spec, attempt_kind, type(err).__name__, str(err)); write_json(actual_path, actual); write_json(d/'actual_outputs_normalization_report.json', actual_result.report)
    if trace_path.exists():
        original_text=trace_path.read_text(errors='ignore')
        if not raw_path.exists(): raw_path.write_text(original_text)
        context=TraceNormalizationContext(task_id=spec.task_id, attempt_id=f'{spec.task_id}_{attempt_kind}', attempt_kind=attempt_kind, task=spec.normalized_instruction, actual_outputs=actual)
        result=repair_agent_trace_file(raw_path, context)
        if result.normalized_trace and (result.report.raw_step_count or result.normalized_trace.get('steps')):
            write_json(d/'agent_trace.normalized.json', result.normalized_trace)
            write_json(d/'agent_trace.json', result.normalized_trace)
            write_json(d/'trace_normalization_report.json', result.report)
            trace = WorkflowTrace.model_validate(result.normalized_trace)
            if recovered_contract_files:
                trace.metadata_json["contract_files_recovered_from"] = recovered_contract_files
                actual.metadata_json.setdefault("contract_files_recovered_from", recovered_contract_files)
            return trace, actual, bool(result.report.trace_schema_valid)
        reason=result.parse_error or 'parseable trace contained no steps'
        final=(d/'final_message.txt').read_text(errors='ignore') if (d/'final_message.txt').exists() else ''
        stderr=(d/'stderr.txt').read_text(errors='ignore') if (d/'stderr.txt').exists() else ''
        msg=f'{reason}\n\nfinal_message:\n{final}\n\nstderr:\n{stderr}'
        trace=_fallback_trace(spec, attempt_kind, prompt, actual, 'TraceNormalizationError', msg, codex_sandbox)
        trace.metadata_json['fallback_reason']=classify_provider_failure(msg).get('fallback_reason') or ('unparseable_trace' if result.fallback_required else 'missing_steps')
        if raw_path.exists():
            raw_path.replace(d/'rejected_agent_response.txt')
        write_json(d/'agent_trace.raw.json', trace)
        write_json(d/'agent_trace.normalized.json', trace)
        write_json(trace_path, trace); _write_fallback_report(d, spec, attempt_kind, trace.metadata_json['fallback_reason'], result.fallback_required)
        return trace, actual, False
    final=(d/'final_message.txt').read_text(errors='ignore') if (d/'final_message.txt').exists() else ''
    stderr=(d/'stderr.txt').read_text(errors='ignore') if (d/'stderr.txt').exists() else ''
    msg=f'missing_trace\n\nfinal_message:\n{final}\n\nstderr:\n{stderr}'
    trace=_fallback_trace(spec, attempt_kind, prompt, actual, 'FileNotFoundError', msg, codex_sandbox)
    trace.metadata_json['fallback_reason']=classify_provider_failure(msg).get('fallback_reason') or 'missing_trace'
    write_json(trace_path, trace); _write_fallback_report(d, spec, attempt_kind, 'missing_trace')
    return trace, actual, False

def deterministic_attempt(package_root: Path, raw: RawTaskRecord, spec: TaskIntakeSpec, attempt_kind='baseline', feedback: str | None=None) -> AttemptResult:
    started_at = utc_now()
    attempt_id=f'{spec.task_id}_{attempt_kind}'
    d=_attempt_dir(package_root, attempt_kind)
    input_files=_copy_attempt_inputs(package_root, d)
    deliverables = _deliverables(raw, spec)
    settings=get_settings()
    prompt=build_worker_prompt(spec.normalized_instruction, '', attempt_kind, input_files, deliverables, settings.sensitive_info_masking, workspace_path=str(d))
    (d/'prompt.md').write_text(prompt)
    (d/'stdout.txt').write_text('deterministic runner completed\n')
    (d/'stderr.txt').write_text('')
    (d/'final_message.txt').write_text(f'Deterministic {attempt_kind} attempt complete.\n')
    files=[]
    for name in deliverables:
        fname=name if '.' in name and '/' not in name else name.lower().replace(' ','_') + '.txt'
        (d/'artifacts'/fname).write_text(f'{attempt_kind} deterministic artifact for {name}\n')
        files.append(f'attempts/{attempt_kind}/artifacts/{fname}')
    actual=ActualOutputs(task_id=spec.task_id, attempt_id=attempt_id, attempt_kind=attempt_kind, status='success', output_summary=f'Deterministic {attempt_kind} outputs for {spec.normalized_title}', primary_output_ref=files[0] if files else None, input_artifact_refs=[f'input/{name}' for name in input_files], deliverable_refs=files, final_message_ref=f'attempts/{attempt_kind}/final_message.txt', artifact_refs=files, files_created=files, files_modified=[], files_deleted=[], stdout_ref=f'attempts/{attempt_kind}/stdout.txt', stderr_ref=f'attempts/{attempt_kind}/stderr.txt', raw_log_refs=[f'attempts/{attempt_kind}/stdout.txt', f'attempts/{attempt_kind}/stderr.txt'], error_type=None, error_message=None, metadata_json={'runner':'deterministic','expected_deliverable_items':deliverables,'produced_deliverable_items':[Path(f).name for f in files]})
    steps=[
        WorkflowTraceStep(step=1, turn=1, actor='user', action='user_message', input=spec.normalized_instruction, message_role='direct_request'),
        WorkflowTraceStep(step=2, turn=1, actor=f'agent:{"reviser" if attempt_kind=="revised" else "worker"}', action='agent_step', operation='plan', tool=None, environment_domain='file', execution_mode='serial', observation='Task instructions, ./input files, and worker-visible rubric are available.', input='Review task and required deliverables.', input_source={'source':'task_packet'}, output='Plan created.', state_change='The attempt plan identified required artifacts.', reasoning='A short plan reduces missed deliverables.', caused_by=[1], causal_type='user_request', success=True, step_outcome='progress', artifact_refs=[f'input/{name}' for name in input_files]),
        WorkflowTraceStep(step=3, turn=1, actor=f'agent:{"reviser" if attempt_kind=="revised" else "worker"}', action='agent_step', operation='write', tool='file_write', environment_domain='file', execution_mode='serial', observation='Deliverable names are known.', input=', '.join(files), input_source={'source':'planned_deliverables'}, output='Artifacts written.', state_change='Required deterministic artifacts were created under ./artifacts.', reasoning='Writing explicit files satisfies the artifact contract.', caused_by=[2], causal_type='execution_of_plan', success=True, step_outcome='completed', artifact_refs=files),
    ]
    ended_at = utc_now()
    trace=WorkflowTrace(trace_id=f'trace_{attempt_id}', collection_id=None, trace_mode='live', task=spec.normalized_instruction, task_id=spec.task_id, attempt_id=attempt_id, attempt_kind=attempt_kind if attempt_kind in ['baseline','revised'] else 'other', agent_tools=['deterministic_runner','file_write'], system_prompt=prompt, system_prompt_hash=None, skills=['agent_trace_skill'], learning='Materialize required artifacts and keep evaluation outside the trace.', termination_reason='task_complete', steps=steps, actual_outputs=actual, artifacts=[], metadata_json={'trace_valid': True})
    trace = _enrich_runtime_trace(trace, actual, package_root, attempt_kind, 'deterministic', started_at, ended_at)
    write_json(d/'actual_outputs.json', actual)
    actual_ctx=ActualOutputsNormalizationContext(task_id=spec.task_id, attempt_id=attempt_id, attempt_kind=attempt_kind, package_root=package_root, required_artifacts=deliverables)
    actual_result=repair_actual_outputs_file(d/'actual_outputs.json', actual_ctx)
    write_actual_outputs_normalization(d, actual_result)
    actual=ActualOutputs.model_validate(actual_result.actual_outputs)
    trace.actual_outputs = actual
    trace = _enrich_runtime_trace(trace, actual, package_root, attempt_kind, 'deterministic', started_at, ended_at)
    write_json(d/'agent_trace.raw.json', trace); write_json(d/'agent_trace.normalized.json', trace); write_json(d/'agent_trace.json', trace)
    report=TraceNormalizationReport(task_id=spec.task_id, attempt_id=attempt_id, attempt_kind=attempt_kind, raw_trace_ref='agent_trace.raw.json', normalized_trace_ref='agent_trace.normalized.json', canonical_trace_ref='agent_trace.json', trace_schema_valid=True, trace_normalized=False, trace_lossless=True, fallback_trace=False, raw_step_count=len(steps), normalized_step_count=len(steps), discarded_step_count=0)
    write_json(d/'trace_normalization_report.json', report)
    return AttemptResult(attempt_dir=str(d), trace_valid=True, trace=trace, actual_outputs=actual)

def codex_command(
    prompt: str,
    sandbox: str,
    workspace: Path | str | None = None,
    *,
    command: str = "codex",
    skip_git_repo_check_supported: bool | None = None,
    ask_for_approval_supported: bool | None = None,
) -> list[str]:
    cmd = [command, 'exec']
    if workspace is not None:
        cmd.extend(['--cd', str(workspace)])
    cmd.extend(['--sandbox', sandbox])
    if ask_for_approval_supported is None:
        ask_for_approval_supported = _codex_exec_supports('--ask-for-approval', command)
    if ask_for_approval_supported:
        cmd.extend(['--ask-for-approval', 'never'])
    if skip_git_repo_check_supported is None:
        skip_git_repo_check_supported = _codex_exec_supports('--skip-git-repo-check', command)
    if skip_git_repo_check_supported:
        cmd.append('--skip-git-repo-check')
    cmd.append(prompt)
    return cmd

def run_codex_attempt(package_root: Path, raw: RawTaskRecord, spec: TaskIntakeSpec, attempt_kind='baseline', timeout=900) -> AttemptResult:
    started_at = utc_now()
    d=_attempt_dir(package_root, attempt_kind)
    input_files=_copy_attempt_inputs(package_root, d)
    deliverables=_deliverables(raw, spec)
    settings=get_settings()
    sandbox=settings.codex_sandbox or 'workspace-write'
    codex_executable=settings.worker_agent_command or 'codex'
    prompt=build_worker_prompt(spec.normalized_instruction, (package_root/'rubric'/'worker_visible_rubric.md').read_text() if (package_root/'rubric'/'worker_visible_rubric.md').exists() else '', attempt_kind, input_files, deliverables, settings.sensitive_info_masking, workspace_path=str(d))
    (d/'prompt.md').write_text(prompt)
    skip_supported=_codex_exec_supports('--skip-git-repo-check', codex_executable)
    ask_supported=_codex_exec_supports('--ask-for-approval', codex_executable)
    cmd=codex_command(
        prompt,
        sandbox,
        d,
        command=codex_executable,
        skip_git_repo_check_supported=skip_supported,
        ask_for_approval_supported=ask_supported,
    )
    run_error=None
    returncode=None
    stdout=''
    stderr=''
    try:
        cp=_run_with_process_group_timeout(cmd, cwd=d, timeout=timeout)
        returncode=cp.returncode
        stdout=cp.stdout or ''
        stderr=cp.stderr or ''
        if cp.returncode != 0 and _is_codex_workspace_trust_error(stdout, stderr) and not skip_supported:
            stderr = f"{stderr}\n{CODEX_TRUST_RETRY_MESSAGE}\nCodex CLI help did not list --skip-git-repo-check, so Agent Apprenticeship could not retry safely."
        elif cp.returncode != 0 and _is_codex_workspace_trust_error(stdout, stderr) and skip_supported and '--skip-git-repo-check' not in cmd:
            stderr = f"{stderr}\n{CODEX_TRUST_RETRY_MESSAGE}"
            retry_cmd=codex_command(
                prompt,
                sandbox,
                d,
                command=codex_executable,
                skip_git_repo_check_supported=True,
                ask_for_approval_supported=ask_supported,
            )
            cp=_run_with_process_group_timeout(retry_cmd, cwd=d, timeout=timeout)
            cmd=retry_cmd
            returncode=cp.returncode
            stdout=(stdout or '') + "\n" + (cp.stdout or '')
            stderr=(stderr or '') + "\n" + (cp.stderr or '')
        (d/'stdout.txt').write_text(redact_secrets(stdout)); (d/'stderr.txt').write_text(redact_secrets(stderr)); (d/'final_message.txt').write_text(redact_secrets((stdout or stderr or '')[-4000:]))
        if cp.returncode != 0:
            run_error=RuntimeError(_apprentice_operational_error(None, stdout, stderr, cp.returncode) or f'Codex exited with code {cp.returncode}.')
    except Exception as e:
        run_error=e
        (d/'stdout.txt').write_text(''); (d/'stderr.txt').write_text(redact_secrets(str(e))); (d/'final_message.txt').write_text('Codex run failed before producing validated trace.')
    recovered_contract_files = _recover_misplaced_contract_files(d)
    contract_diagnostics = None
    if not (d/'agent_trace.json').exists() or not (d/'actual_outputs.json').exists():
        contract_diagnostics = build_contract_diagnostics(d, command=cmd, working_directory=d, agent_display_name='Codex', prompt=prompt)
        with (d/'final_message.txt').open('a') as f:
            f.write('\n\n' + diagnostics_text(contract_diagnostics))
    trace, actual, trace_valid = ensure_attempt_outputs(package_root, spec, attempt_kind, prompt, sandbox, run_error)
    ended_at = utc_now()
    trace = _enrich_runtime_trace(trace, actual, package_root, attempt_kind, 'codex', started_at, ended_at)
    trace.metadata_json['codex_sandbox']=sandbox
    trace.metadata_json['codex_command']=[part if part != prompt else '<prompt>' for part in cmd]
    trace.metadata_json['codex_skip_git_repo_check_supported']=skip_supported
    trace.metadata_json['codex_skip_git_repo_check_used']='--skip-git-repo-check' in cmd
    trace.metadata_json['trace_valid']=trace_valid
    if recovered_contract_files:
        trace.metadata_json['contract_files_recovered_from'] = recovered_contract_files
    if contract_diagnostics:
        trace.metadata_json['apprentice_agent_contract_diagnostics']=contract_diagnostics
    write_json(d/'agent_trace.json', trace)
    if actual.metadata_json is None: actual.metadata_json={}
    actual.metadata_json['codex_sandbox']=sandbox
    if recovered_contract_files:
        actual.metadata_json['contract_files_recovered_from'] = recovered_contract_files
    if contract_diagnostics:
        actual.metadata_json['apprentice_agent_contract_diagnostics']=contract_diagnostics
    op_error = _apprentice_operational_error(run_error, stdout, stderr, returncode)
    if (
        op_error
        and isinstance(run_error, subprocess.TimeoutExpired)
        and trace_valid
        and actual.status == 'success'
    ):
        actual.metadata_json['apprentice_agent_warning'] = (
            'Apprentice Agent process timed out after producing required outputs; '
            'the produced trace and artifacts were preserved.'
        )
        trace.metadata_json['apprentice_agent_warning'] = actual.metadata_json['apprentice_agent_warning']
        op_error = None
    if op_error and returncode not in (None, 0) and trace_valid and actual.status == 'success':
        op_error = f"Apprentice Agent exited nonzero after producing required outputs (exit code {returncode})."
    if op_error or not trace_valid:
        actual.metadata_json['apprentice_agent_operational_error'] = op_error or 'Apprentice Agent did not produce a valid agent_trace.json; raw logs were preserved.'
        actual.error_message = op_error or actual.error_message
        trace.metadata_json['apprentice_agent_operational_error'] = actual.metadata_json['apprentice_agent_operational_error']
    write_json(d/'actual_outputs.json', actual)
    write_json(d/'agent_trace.json', trace)
    return AttemptResult(attempt_dir=str(d), trace_valid=trace_valid, trace=trace, actual_outputs=actual, codex_sandbox=sandbox)


def run_custom_attempt(package_root: Path, raw: RawTaskRecord, spec: TaskIntakeSpec, attempt_kind='baseline', timeout: int | None=None) -> AttemptResult:
    started_at = utc_now()
    settings=get_settings()
    d=_attempt_dir(package_root, attempt_kind)
    input_files=_copy_attempt_inputs(package_root, d)
    deliverables=_deliverables(raw, spec)
    rubric_md=(package_root/'rubric'/'worker_visible_rubric.md').read_text() if (package_root/'rubric'/'worker_visible_rubric.md').exists() else ''
    prompt=build_worker_prompt(spec.normalized_instruction, rubric_md, attempt_kind, input_files, deliverables, settings.sensitive_info_masking, workspace_path=str(d))
    prompt_file=d/'prompt.md'
    prompt_file.write_text(prompt)
    template=settings.custom_worker_command_template
    if not template:
        err=RuntimeError('Custom Apprentice Agent is configured without a command template.')
        (d/'stdout.txt').write_text('')
        (d/'stderr.txt').write_text(str(err))
        (d/'final_message.txt').write_text('Custom Apprentice Agent configuration error.')
        contract_diagnostics = build_contract_diagnostics(d, command='custom-agent', working_directory=d, agent_display_name=settings.custom_worker_display_name or 'Custom', prompt=prompt)
        with (d/'final_message.txt').open('a') as f:
            f.write('\n\n' + diagnostics_text(contract_diagnostics))
        trace, actual, trace_valid=ensure_attempt_outputs(package_root, spec, attempt_kind, prompt, 'custom', err)
        ended_at = utc_now()
        trace = _enrich_runtime_trace(trace, actual, package_root, attempt_kind, 'custom', started_at, ended_at)
        if actual.metadata_json is None:
            actual.metadata_json = {}
        actual.metadata_json['apprentice_agent_contract_diagnostics'] = contract_diagnostics
        write_json(d/'actual_outputs.json', actual)
        trace.metadata_json['apprentice_agent_contract_diagnostics'] = contract_diagnostics
        write_json(d/'agent_trace.json', trace)
        return AttemptResult(attempt_dir=str(d), trace_valid=trace_valid, trace=trace, actual_outputs=actual, custom_worker_error=str(err))
    replacements={
        'workspace': str(d),
        'prompt_file': str(prompt_file),
        'run_dir': str(package_root.parent.parent),
        'task_instruction': spec.normalized_instruction,
    }
    command=template
    for key, value in replacements.items():
        command=command.replace('{'+key+'}', value)
    run_error=None
    cp=None
    try:
        cp=_run_with_process_group_timeout(command, cwd=d, timeout=timeout or settings.task_timeout_seconds, shell=True)
        stdout=redact_secrets(cp.stdout or '')
        stderr=redact_secrets(cp.stderr or '')
        (d/'stdout.txt').write_text(stdout)
        (d/'stderr.txt').write_text(stderr)
        (d/'final_message.txt').write_text(redact_secrets((cp.stdout or cp.stderr or '')[-4000:]))
        if cp.returncode != 0:
            run_error=RuntimeError(f'Apprentice Agent exited before producing required outputs (exit code {cp.returncode}).')
    except Exception as e:
        run_error=e
        (d/'stdout.txt').write_text('')
        (d/'stderr.txt').write_text(redact_secrets(str(e)))
        (d/'final_message.txt').write_text('Custom Apprentice Agent failed before producing validated trace.')
    contract_diagnostics = None
    if not (d/'agent_trace.json').exists() or not (d/'actual_outputs.json').exists():
        contract_diagnostics = build_contract_diagnostics(d, command=command, working_directory=d, agent_display_name=settings.custom_worker_display_name or 'Custom', prompt=prompt)
        with (d/'final_message.txt').open('a') as f:
            f.write('\n\n' + diagnostics_text(contract_diagnostics))
    trace, actual, trace_valid=ensure_attempt_outputs(package_root, spec, attempt_kind, prompt, 'custom', run_error)
    ended_at = utc_now()
    trace = _enrich_runtime_trace(trace, actual, package_root, attempt_kind, 'custom', started_at, ended_at)
    trace.metadata_json['custom_worker_display_name']=settings.custom_worker_display_name
    trace.metadata_json['custom_worker_command_template']=template
    trace.metadata_json['trace_valid']=trace_valid
    if contract_diagnostics:
        trace.metadata_json['apprentice_agent_contract_diagnostics']=contract_diagnostics
    write_json(d/'agent_trace.json', trace)
    if actual.metadata_json is None: actual.metadata_json={}
    actual.metadata_json['apprentice_agent']='custom'
    if contract_diagnostics:
        actual.metadata_json['apprentice_agent_contract_diagnostics']=contract_diagnostics
    if run_error:
        actual.metadata_json['apprentice_agent_operational_error']=str(run_error)
    write_json(d/'actual_outputs.json', actual)
    return AttemptResult(
        attempt_dir=str(d),
        trace_valid=trace_valid,
        trace=trace,
        actual_outputs=actual,
        custom_worker_returncode=(cp.returncode if cp else None),
    )
