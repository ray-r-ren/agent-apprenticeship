from __future__ import annotations

import os
import shlex
import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .env import load_local_env
from .command_discovery import resolve_command, resolve_agent_command, gui_app_hint
from .io import read_workflow_json as read_json, write_json
from .recipes import MODEL_PROVIDER_RECIPES, WORKER_AGENT_RECIPES


ApprenticeshipMode = Literal["autonomous", "expert_led", "org_custom"]
MentorMode = Literal["model_assisted", "expert_led", "hybrid"]
SensitiveInfoMasking = Literal["standard", "no_masking"]
EcosystemAutoShare = Literal["manual", "automatic", "disabled"]
TrainingContributionMode = Literal["public_ecosystem", "private_internal_only"]
EcosystemStorageBackend = Literal["forsy-r2", "github"]
EvaluationMode = Literal["model-assisted", "expert-led", "hybrid"]
DataSharingLevel = Literal["standard", "full-context"]

APPRENTICESHIP_MODES: tuple[str, ...] = ("autonomous", "expert_led", "org_custom")
MENTOR_MODES: tuple[str, ...] = ("model_assisted", "expert_led", "hybrid")
SENSITIVE_INFO_MASKING_LEVELS: tuple[str, ...] = ("standard", "no_masking")
ECOSYSTEM_AUTO_SHARE_MODES: tuple[str, ...] = ("manual", "automatic", "disabled")
TRAINING_CONTRIBUTION_MODES: tuple[str, ...] = ("public_ecosystem", "private_internal_only")
EVALUATION_MODES: tuple[str, ...] = ("model-assisted", "expert-led", "hybrid")
DATA_SHARING_LEVELS: tuple[str, ...] = ("standard", "full-context")
DEFAULT_APP_HOME = Path("~/.agent-apprenticeship").expanduser()
DEFAULT_PUBLIC_ECOSYSTEM_REPO = "Forsy-AI/agent-apprenticeship"
DEFAULT_PUBLIC_ECOSYSTEM_URL = f"https://github.com/{DEFAULT_PUBLIC_ECOSYSTEM_REPO}"
DEFAULT_ECOSYSTEM_STORAGE_BACKEND = "forsy-r2"
DEFAULT_R2_BUCKET = "apprenticeship-dev"
DEFAULT_R2_INDEX_PREFIX = "indexes"
DEFAULT_R2_CONTRIBUTION_PREFIX = "contributions/packages"
DEFAULT_R2_SEED_PREFIX = "contributions/seed_dataset"
DEFAULT_R2_API_BASE = "https://agent-apprenticeship-ingest.sai-1c1.workers.dev"
_SETTINGS_OVERRIDE: ContextVar[Any] = ContextVar("agent_apprenticeship_settings_override", default=None)


def normalize_apprenticeship_mode(value: str | None, default: str = "autonomous") -> str:
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "autonomous": "autonomous",
        "autonomous_apprenticeship": "autonomous",
        "model": "autonomous",
        "model_assisted": "autonomous",
        "model_assisted_automated": "autonomous",
        "modelassisted": "autonomous",
        "automated": "autonomous",
        "expert": "expert_led",
        "expert_led": "expert_led",
        "expert_led_apprenticeship": "expert_led",
        "expert_led_manual": "expert_led",
        "manual": "expert_led",
        "human": "expert_led",
        "hybrid": "expert_led",
        "org": "org_custom",
        "organization": "org_custom",
        "organization_custom": "org_custom",
        "org_custom": "org_custom",
        "custom_org": "org_custom",
        "custom": "org_custom",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in APPRENTICESHIP_MODES:
        raise ValueError(f"Unsupported Apprenticeship Mode: {value}")
    return normalized


def apprenticeship_mode_display(value: str | None) -> str:
    normalized = normalize_apprenticeship_mode(value)
    return {
        "autonomous": "Autonomous Apprenticeship",
        "expert_led": "Expert-Led Apprenticeship",
        "org_custom": "Organization Custom",
    }[normalized]


def apprenticeship_mode_to_mentor_mode(value: str | None) -> str:
    normalized = normalize_apprenticeship_mode(value)
    return {
        "autonomous": "model_assisted",
        "expert_led": "expert_led",
        "org_custom": "expert_led",
    }[normalized]


def mentor_mode_to_apprenticeship_mode(value: str | None) -> str:
    normalized = normalize_mentor_mode(value)
    return {
        "model_assisted": "autonomous",
        "expert_led": "expert_led",
        "hybrid": "expert_led",
    }[normalized]


def normalize_mentor_mode(value: str | None, default: str = "model_assisted") -> str:
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "model": "model_assisted",
        "model_assisted": "model_assisted",
        "model_assisted_automated": "model_assisted",
        "automated": "model_assisted",
        "expert": "expert_led",
        "expert_led": "expert_led",
        "expert_led_manual": "expert_led",
        "manual": "expert_led",
        "hybrid": "hybrid",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MENTOR_MODES:
        raise ValueError(f"Unsupported apprenticeship mode: {value}")
    return normalized


def mentor_mode_to_evaluation_mode(value: str | None) -> str:
    normalized = normalize_mentor_mode(value)
    return {
        "model_assisted": "model-assisted",
        "expert_led": "expert-led",
        "hybrid": "hybrid",
    }[normalized]


def evaluation_mode_to_mentor_mode(value: str | None) -> str:
    return normalize_mentor_mode(value)


def mentor_mode_display(value: str | None) -> str:
    return apprenticeship_mode_display(mentor_mode_to_apprenticeship_mode(value))


def normalize_sensitive_info_masking(value: str | None, default: str = "standard") -> str:
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "standard": "standard",
        "mask": "standard",
        "masked": "standard",
        "full": "no_masking",
        "full_context": "no_masking",
        "no_masking": "no_masking",
        "none": "no_masking",
        "off": "no_masking",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SENSITIVE_INFO_MASKING_LEVELS:
        raise ValueError(f"Unsupported Sensitive Info Masking setting: {value}")
    return normalized


def sensitive_info_masking_to_data_sharing(value: str | None) -> str:
    return "standard" if normalize_sensitive_info_masking(value) == "standard" else "full-context"


def data_sharing_to_sensitive_info_masking(value: str | None) -> str:
    return normalize_sensitive_info_masking(value)


def normalize_ecosystem_auto_share(value: str | None, default: str = "manual") -> str:
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "manual": "manual",
        "ask": "manual",
        "prompt": "manual",
        "ask_before_sharing": "manual",
        "share_automatically": "automatic",
        "automatic": "automatic",
        "auto": "automatic",
        "on": "automatic",
        "disabled": "disabled",
        "disable": "disabled",
        "off": "disabled",
        "none": "disabled",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ECOSYSTEM_AUTO_SHARE_MODES:
        raise ValueError(f"Unsupported ecosystem sharing setting: {value}")
    return normalized


def ecosystem_auto_share_display(value: str | None) -> str:
    normalized = normalize_ecosystem_auto_share(value)
    return {
        "manual": "manual",
        "automatic": "automatic",
        "disabled": "disabled",
    }[normalized]


def normalize_training_contribution_mode(value: str | None, default: str = "public_ecosystem") -> str:
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "public": "public_ecosystem",
        "public_ecosystem": "public_ecosystem",
        "public_ecosystem_default": "public_ecosystem",
        "ecosystem": "public_ecosystem",
        "share": "public_ecosystem",
        "shared": "public_ecosystem",
        "automatic": "public_ecosystem",
        "auto": "public_ecosystem",
        "on": "public_ecosystem",
        "private": "private_internal_only",
        "private_internal": "private_internal_only",
        "private_internal_only": "private_internal_only",
        "internal": "private_internal_only",
        "local": "private_internal_only",
        "manual": "private_internal_only",
        "ask": "private_internal_only",
        "disabled": "private_internal_only",
        "disable": "private_internal_only",
        "off": "private_internal_only",
        "none": "private_internal_only",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TRAINING_CONTRIBUTION_MODES:
        raise ValueError(f"Unsupported Agent Training Contribution Mode: {value}")
    return normalized


def training_contribution_mode_display(value: str | None) -> str:
    normalized = normalize_training_contribution_mode(value)
    return {
        "public_ecosystem": "Public Ecosystem",
        "private_internal_only": "Private Internal Only",
    }[normalized]


def training_contribution_to_auto_share(value: str | None) -> str:
    return "automatic" if normalize_training_contribution_mode(value) == "public_ecosystem" else "disabled"


def auto_share_to_training_contribution(value: str | None) -> str:
    return "private_internal_only" if normalize_ecosystem_auto_share(value) == "disabled" else "public_ecosystem"


def normalize_ecosystem_storage(value: str | None, default: str = DEFAULT_ECOSYSTEM_STORAGE_BACKEND) -> str:
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "r2": "forsy-r2",
        "cloudflare-r2": "forsy-r2",
        "cloudflare_r2": "forsy-r2",
        "forsy-r2": "forsy-r2",
        "github": "github",
        "github-issue": "github",
        "legacy-github": "github",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"forsy-r2", "github"}:
        raise ValueError(f"Unsupported ecosystem storage backend: {value}")
    return normalized


class Settings(BaseModel):
    # Public product settings.
    app_home: Path = DEFAULT_APP_HOME
    worker_agent: str = "codex"
    worker_agent_command: str | None = None
    worker_agent_model: str | None = None
    worker_agent_extra_args: list[str] = Field(default_factory=list)
    model_provider: str | None = None
    model_provider_api_key_env: str | None = None
    model_provider_model: str | None = None
    apprenticeship_mode: ApprenticeshipMode = "autonomous"
    mentor_mode: MentorMode = "model_assisted"
    sensitive_info_masking: SensitiveInfoMasking = "standard"
    evaluation_mode: EvaluationMode = "model-assisted"
    data_sharing_level: DataSharingLevel = "standard"
    max_improvement_loops: int = 5
    custom_worker_display_name: str | None = None
    custom_worker_command_template: str | None = None
    custom_worker_can_write_files: bool = True
    apprentice_agent_readiness_status: str | None = None
    apprentice_agent_readiness_reason: str | None = None
    mentor_model_provider_readiness_status: str | None = None
    mentor_model_provider_readiness_reason: str | None = None
    ecosystem_repo: str | None = DEFAULT_PUBLIC_ECOSYSTEM_REPO
    ecosystem_repo_path: Path | None = None
    training_contribution_mode: TrainingContributionMode = "public_ecosystem"
    ecosystem_auto_share: EcosystemAutoShare = "automatic"
    ecosystem_storage_backend: EcosystemStorageBackend = DEFAULT_ECOSYSTEM_STORAGE_BACKEND
    r2_api_url: str | None = DEFAULT_R2_API_BASE
    r2_bucket: str = DEFAULT_R2_BUCKET
    r2_index_prefix: str = DEFAULT_R2_INDEX_PREFIX
    r2_contribution_prefix: str = DEFAULT_R2_CONTRIBUTION_PREFIX
    r2_seed_prefix: str = DEFAULT_R2_SEED_PREFIX

    # Compatibility settings used by the existing internal pipeline.
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"
    worker_runner: str = "codex"
    reviser_runner: str = "codex"
    allow_deterministic_fallback: bool = True
    task_timeout_seconds: int = 900
    max_iterations: int = 3
    codex_sandbox: str = "workspace-write"
    llm_evaluator_enabled: bool = True
    llm_grader_enabled: bool = True
    llm_verifier_enabled: bool = True
    llm_task_intake_model: str = "gpt-5-mini"
    llm_rubric_model: str = "gpt-5-mini"
    llm_evaluator_model: str = "gpt-5-mini"
    llm_grader_model: str = "gpt-5-mini"
    llm_verifier_model: str = "gpt-5-mini"
    llm_judge_count: int = 1
    llm_fail_closed: bool = False
    allow_deterministic_eval_fallback: bool = True
    rubric_mode: str = "hybrid"
    llm_task_intake_enabled: bool = True
    llm_rubric_generation_enabled: bool = True


@contextmanager
def settings_override(settings: Settings):
    token = _SETTINGS_OVERRIDE.set(settings)
    try:
        yield
    finally:
        _SETTINGS_OVERRIDE.reset(token)


def app_home_from_env() -> Path:
    return Path(os.getenv("AA_HOME") or DEFAULT_APP_HOME).expanduser()


def settings_path(app_home: Path | None = None) -> Path:
    return (app_home or app_home_from_env()) / "settings.json"


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() == "true"


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _stored_settings() -> dict[str, Any]:
    # Test suites and CI often set AA_DISABLE_LOCAL_ENV to avoid reading user
    # machine state. Honor that unless AA_HOME intentionally points at a test dir.
    if os.getenv("AA_DISABLE_LOCAL_ENV") == "1" and not os.getenv("AA_HOME"):
        return {}
    path = settings_path()
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def default_settings(app_home: Path | None = None) -> Settings:
    home = app_home or app_home_from_env()
    return Settings(app_home=home, max_iterations=3)


def get_settings(root: Path | None = None) -> Settings:
    if root is None:
        override = _SETTINGS_OVERRIDE.get()
        if override is not None:
            return override
    load_local_env(root)
    stored = _stored_settings()
    home = Path(os.getenv("AA_HOME") or stored.get("app_home") or DEFAULT_APP_HOME).expanduser()
    load_local_env(home)

    worker_agent = os.getenv("AA_WORKER_AGENT") or os.getenv("AA_TRACE_WORKER_AGENT") or stored.get("worker_agent") or "codex"
    provider = os.getenv("AA_MODEL_PROVIDER") or stored.get("model_provider")
    provider_recipe = MODEL_PROVIDER_RECIPES.get(str(provider)) if provider else None
    provider_key_env = (
        os.getenv("AA_MODEL_PROVIDER_API_KEY_ENV")
        or stored.get("model_provider_api_key_env")
        or (provider_recipe.api_key_env_var if provider_recipe else None)
    )
    provider_model = (
        os.getenv("AA_MODEL_PROVIDER_MODEL")
        or stored.get("model_provider_model")
        or (provider_recipe.default_model if provider_recipe else None)
    )

    loops = _int_env(
        "AA_MAX_IMPROVEMENT_LOOPS",
        int(stored.get("max_improvement_loops") or stored.get("max_iterations") or 5),
    )
    legacy_iterations = _int_env("AA_MAX_ITERATIONS", int(stored.get("max_iterations") or 3))
    openai_model = os.getenv("AA_OPENAI_MODEL", stored.get("openai_model") or "gpt-5-mini")
    new_mode_source = os.getenv("AA_APPRENTICESHIP_MODE") or stored.get("apprenticeship_mode")
    legacy_mode_source = (
        os.getenv("AA_MENTOR_MODE")
        or stored.get("mentor_mode")
        or os.getenv("AA_EVALUATION_MODE")
        or stored.get("evaluation_mode")
    )
    apprenticeship_mode = normalize_apprenticeship_mode(new_mode_source or legacy_mode_source or "autonomous")
    mentor_mode = apprenticeship_mode_to_mentor_mode(apprenticeship_mode)
    if not new_mode_source and legacy_mode_source:
        legacy_mentor_mode = normalize_mentor_mode(legacy_mode_source)
        if legacy_mentor_mode == "hybrid":
            mentor_mode = "hybrid"
    sensitive_info_masking = normalize_sensitive_info_masking(
        os.getenv("AA_SENSITIVE_INFO_MASKING")
        or stored.get("sensitive_info_masking")
        or os.getenv("AA_DATA_SHARING_LEVEL")
        or stored.get("data_sharing_level")
        or "standard"
    )
    auto_share_source = os.getenv("AA_ECOSYSTEM_AUTO_SHARE") or stored.get("ecosystem_auto_share")
    training_contribution_source = os.getenv("AA_TRAINING_CONTRIBUTION_MODE") or stored.get("training_contribution_mode")
    if training_contribution_source:
        training_contribution_mode = normalize_training_contribution_mode(training_contribution_source)
    elif auto_share_source:
        training_contribution_mode = auto_share_to_training_contribution(auto_share_source)
    else:
        training_contribution_mode = "public_ecosystem"
    if auto_share_source:
        ecosystem_auto_share = normalize_ecosystem_auto_share(auto_share_source)
    else:
        ecosystem_auto_share = training_contribution_to_auto_share(training_contribution_mode)

    return Settings(
        app_home=home,
        worker_agent=worker_agent,
        worker_agent_command=os.getenv("AA_WORKER_AGENT_COMMAND") or stored.get("worker_agent_command"),
        worker_agent_model=os.getenv("AA_WORKER_AGENT_MODEL") or stored.get("worker_agent_model"),
        worker_agent_extra_args=list(stored.get("worker_agent_extra_args") or []),
        model_provider=provider,
        model_provider_api_key_env=provider_key_env,
        model_provider_model=provider_model,
        apprenticeship_mode=apprenticeship_mode,
        mentor_mode=mentor_mode,
        sensitive_info_masking=sensitive_info_masking,
        evaluation_mode=mentor_mode_to_evaluation_mode(mentor_mode),
        data_sharing_level=sensitive_info_masking_to_data_sharing(sensitive_info_masking),
        max_improvement_loops=loops,
        custom_worker_display_name=os.getenv("AA_CUSTOM_WORKER_DISPLAY_NAME") or stored.get("custom_worker_display_name"),
        custom_worker_command_template=os.getenv("AA_CUSTOM_WORKER_COMMAND_TEMPLATE") or stored.get("custom_worker_command_template"),
        custom_worker_can_write_files=_bool_env("AA_CUSTOM_WORKER_CAN_WRITE_FILES", bool(stored.get("custom_worker_can_write_files", True))),
        apprentice_agent_readiness_status=stored.get("apprentice_agent_readiness_status"),
        apprentice_agent_readiness_reason=stored.get("apprentice_agent_readiness_reason"),
        mentor_model_provider_readiness_status=stored.get("mentor_model_provider_readiness_status"),
        mentor_model_provider_readiness_reason=stored.get("mentor_model_provider_readiness_reason"),
        ecosystem_repo=os.getenv("AA_ECOSYSTEM_REPO") or stored.get("ecosystem_repo") or DEFAULT_PUBLIC_ECOSYSTEM_REPO,
        ecosystem_repo_path=Path(os.getenv("AA_ECOSYSTEM_REPO_PATH") or stored.get("ecosystem_repo_path")).expanduser()
        if (os.getenv("AA_ECOSYSTEM_REPO_PATH") or stored.get("ecosystem_repo_path"))
        else None,
        training_contribution_mode=training_contribution_mode,
        ecosystem_auto_share=ecosystem_auto_share,
        ecosystem_storage_backend=normalize_ecosystem_storage(
            os.getenv("AA_R2_STORAGE_BACKEND")
            or os.getenv("AA_ECOSYSTEM_STORAGE_BACKEND")
            or stored.get("ecosystem_storage_backend")
            or DEFAULT_ECOSYSTEM_STORAGE_BACKEND
        ),
        r2_api_url=(
            os.getenv("AA_R2_API_BASE")
            or os.getenv("AA_R2_API_URL")
            or stored.get("r2_api_url")
            or DEFAULT_R2_API_BASE
        ),
        r2_bucket=os.getenv("AA_R2_BUCKET") or stored.get("r2_bucket") or DEFAULT_R2_BUCKET,
        r2_index_prefix=os.getenv("AA_R2_INDEX_PREFIX") or stored.get("r2_index_prefix") or DEFAULT_R2_INDEX_PREFIX,
        r2_contribution_prefix=os.getenv("AA_R2_CONTRIBUTION_PREFIX") or stored.get("r2_contribution_prefix") or DEFAULT_R2_CONTRIBUTION_PREFIX,
        r2_seed_prefix=os.getenv("AA_R2_SEED_PREFIX") or stored.get("r2_seed_prefix") or DEFAULT_R2_SEED_PREFIX,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=openai_model,
        worker_runner=os.getenv("AA_WORKER_RUNNER", os.getenv("AA_TRACE_WORKER_RUNNER", stored.get("worker_runner") or worker_agent)),
        reviser_runner=os.getenv("AA_REVISER_RUNNER", os.getenv("AA_TRACE_REVISER_RUNNER", stored.get("reviser_runner") or worker_agent)),
        allow_deterministic_fallback=_bool_env("AA_ALLOW_DETERMINISTIC_FALLBACK", bool(stored.get("allow_deterministic_fallback", True))),
        task_timeout_seconds=_int_env("AA_TASK_TIMEOUT_SECONDS", int(stored.get("task_timeout_seconds") or 900)),
        max_iterations=legacy_iterations,
        codex_sandbox=os.getenv("AA_CODEX_SANDBOX", stored.get("codex_sandbox") or "workspace-write"),
        llm_evaluator_enabled=_bool_env("AA_LLM_EVALUATOR_ENABLED", bool(stored.get("llm_evaluator_enabled", True))),
        llm_grader_enabled=_bool_env("AA_LLM_GRADER_ENABLED", bool(stored.get("llm_grader_enabled", True))),
        llm_verifier_enabled=_bool_env("AA_LLM_VERIFIER_ENABLED", bool(stored.get("llm_verifier_enabled", True))),
        llm_task_intake_model=os.getenv("AA_LLM_TASK_INTAKE_MODEL", stored.get("llm_task_intake_model") or openai_model),
        llm_rubric_model=os.getenv("AA_LLM_RUBRIC_MODEL", stored.get("llm_rubric_model") or openai_model),
        llm_evaluator_model=os.getenv("AA_LLM_EVALUATOR_MODEL", stored.get("llm_evaluator_model") or openai_model),
        llm_grader_model=os.getenv("AA_LLM_GRADER_MODEL", stored.get("llm_grader_model") or openai_model),
        llm_verifier_model=os.getenv("AA_LLM_VERIFIER_MODEL", stored.get("llm_verifier_model") or openai_model),
        llm_judge_count=_int_env("AA_LLM_JUDGE_COUNT", int(stored.get("llm_judge_count") or 1)),
        llm_fail_closed=_bool_env("AA_LLM_FAIL_CLOSED", bool(stored.get("llm_fail_closed", False))),
        allow_deterministic_eval_fallback=_bool_env("AA_ALLOW_DETERMINISTIC_EVAL_FALLBACK", bool(stored.get("allow_deterministic_eval_fallback", True))),
        rubric_mode=os.getenv("AA_RUBRIC_MODE", stored.get("rubric_mode") or "hybrid"),
        llm_task_intake_enabled=_bool_env("AA_LLM_TASK_INTAKE_ENABLED", bool(stored.get("llm_task_intake_enabled", True))),
        llm_rubric_generation_enabled=_bool_env("AA_LLM_RUBRIC_GENERATION_ENABLED", bool(stored.get("llm_rubric_generation_enabled", True))),
    )


def save_settings(settings: Settings) -> Path:
    settings.app_home.mkdir(parents=True, exist_ok=True)
    path = settings_path(settings.app_home)
    data = settings.model_dump(mode="json")
    data.pop("openai_api_key", None)
    write_json(path, data)
    return path


def init_settings(app_home: Path | None = None, overwrite: bool = False) -> Path:
    home = app_home or app_home_from_env()
    home.mkdir(parents=True, exist_ok=True)
    (home / "runs").mkdir(exist_ok=True)
    path = settings_path(home)
    if overwrite or not path.exists():
        save_settings(default_settings(home))
    return path


def update_settings(**updates: Any) -> Settings:
    settings = get_settings()
    data = settings.model_dump()
    clean_updates = {k: v for k, v in updates.items() if v is not None}
    legacy_requested_mode = None
    if "apprenticeship_mode" not in clean_updates:
        if "mentor_mode" in clean_updates:
            legacy_requested_mode = clean_updates["mentor_mode"]
        elif "evaluation_mode" in clean_updates:
            legacy_requested_mode = evaluation_mode_to_mentor_mode(clean_updates["evaluation_mode"])
    if "mentor_mode" in clean_updates and "apprenticeship_mode" not in clean_updates:
        clean_updates["apprenticeship_mode"] = mentor_mode_to_apprenticeship_mode(clean_updates["mentor_mode"])
    if "evaluation_mode" in clean_updates and "mentor_mode" not in clean_updates:
        clean_updates["apprenticeship_mode"] = mentor_mode_to_apprenticeship_mode(
            evaluation_mode_to_mentor_mode(clean_updates["evaluation_mode"])
        )
    if "apprenticeship_mode" in clean_updates:
        clean_updates["apprenticeship_mode"] = normalize_apprenticeship_mode(clean_updates["apprenticeship_mode"])
        clean_updates["mentor_mode"] = apprenticeship_mode_to_mentor_mode(clean_updates["apprenticeship_mode"])
        clean_updates["evaluation_mode"] = mentor_mode_to_evaluation_mode(clean_updates["mentor_mode"])
    if "mentor_mode" in clean_updates:
        clean_updates["mentor_mode"] = normalize_mentor_mode(clean_updates["mentor_mode"])
        clean_updates["apprenticeship_mode"] = mentor_mode_to_apprenticeship_mode(clean_updates["mentor_mode"])
        clean_updates["evaluation_mode"] = mentor_mode_to_evaluation_mode(clean_updates["mentor_mode"])
    if legacy_requested_mode and normalize_mentor_mode(legacy_requested_mode) == "hybrid":
        clean_updates["apprenticeship_mode"] = "expert_led"
        clean_updates["mentor_mode"] = "hybrid"
        clean_updates["evaluation_mode"] = "hybrid"
    if "data_sharing_level" in clean_updates and "sensitive_info_masking" not in clean_updates:
        clean_updates["sensitive_info_masking"] = data_sharing_to_sensitive_info_masking(clean_updates["data_sharing_level"])
    if "sensitive_info_masking" in clean_updates:
        clean_updates["sensitive_info_masking"] = normalize_sensitive_info_masking(clean_updates["sensitive_info_masking"])
        clean_updates["data_sharing_level"] = sensitive_info_masking_to_data_sharing(clean_updates["sensitive_info_masking"])
    if "ecosystem_auto_share" in clean_updates and "training_contribution_mode" not in clean_updates:
        clean_updates["training_contribution_mode"] = auto_share_to_training_contribution(clean_updates["ecosystem_auto_share"])
    if "training_contribution_mode" in clean_updates:
        clean_updates["training_contribution_mode"] = normalize_training_contribution_mode(clean_updates["training_contribution_mode"])
        if "ecosystem_auto_share" not in clean_updates:
            clean_updates["ecosystem_auto_share"] = training_contribution_to_auto_share(clean_updates["training_contribution_mode"])
    if "ecosystem_auto_share" in clean_updates:
        clean_updates["ecosystem_auto_share"] = normalize_ecosystem_auto_share(clean_updates["ecosystem_auto_share"])
    if "ecosystem_storage_backend" in clean_updates:
        clean_updates["ecosystem_storage_backend"] = normalize_ecosystem_storage(clean_updates["ecosystem_storage_backend"])
    if "worker_agent" in clean_updates and "worker_runner" not in clean_updates:
        clean_updates["worker_runner"] = clean_updates["worker_agent"]
        clean_updates["reviser_runner"] = clean_updates["worker_agent"]
    data.update(clean_updates)
    updated = Settings.model_validate(data)
    if updated.worker_agent not in WORKER_AGENT_RECIPES:
        raise ValueError(f"Unsupported Apprentice Agent: {updated.worker_agent}")
    if updated.model_provider is not None and updated.model_provider not in MODEL_PROVIDER_RECIPES:
        raise ValueError(f"Unsupported Mentor Model Provider: {updated.model_provider}")
    if updated.evaluation_mode not in EVALUATION_MODES:
        raise ValueError(f"Unsupported Apprenticeship Mode: {updated.apprenticeship_mode}")
    if updated.data_sharing_level not in DATA_SHARING_LEVELS:
        raise ValueError(f"Unsupported Sensitive Info Masking setting: {updated.sensitive_info_masking}")
    if updated.mentor_mode not in MENTOR_MODES:
        raise ValueError(f"Unsupported legacy apprenticeship mode setting: {updated.mentor_mode}")
    if updated.apprenticeship_mode not in APPRENTICESHIP_MODES:
        raise ValueError(f"Unsupported Apprenticeship Mode: {updated.apprenticeship_mode}")
    if updated.sensitive_info_masking not in SENSITIVE_INFO_MASKING_LEVELS:
        raise ValueError(f"Unsupported sensitive info masking setting: {updated.sensitive_info_masking}")
    if updated.ecosystem_auto_share not in ECOSYSTEM_AUTO_SHARE_MODES:
        raise ValueError(f"Unsupported ecosystem sharing setting: {updated.ecosystem_auto_share}")
    if updated.training_contribution_mode not in TRAINING_CONTRIBUTION_MODES:
        raise ValueError(f"Unsupported Agent Training Contribution Mode: {updated.training_contribution_mode}")
    if updated.worker_agent == "custom":
        template = updated.custom_worker_command_template or ""
        missing = [token for token in ("{workspace}", "{prompt_file}") if token not in template]
        if missing:
            raise ValueError(
                "custom Apprentice Agent command template must include placeholders: " + ", ".join(missing)
            )
    if updated.max_improvement_loops < 1:
        raise ValueError("max_improvement_loops must be at least 1")
    updated.max_iterations = updated.max_improvement_loops
    save_settings(updated)
    return updated


def model_provider_ready(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    if s.mentor_mode == "expert_led":
        return True
    return bool(mentor_model_provider_readiness(s).get("ready"))


def configured_model_provider_ready(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return bool(mentor_model_provider_readiness(s).get("ready"))


def model_provider_display_name(settings: Settings | None = None) -> str:
    s = settings or get_settings()
    provider = MODEL_PROVIDER_RECIPES.get(s.model_provider or "")
    return provider.display_name if provider else "Not configured"


def mentor_model_provider_readiness(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    if not s.model_provider:
        return {
            "status": "not_ready",
            "ready": False,
            "reason": None,
            "provider": None,
            "provider_display": "Not configured",
            "model": None,
            "api_key_env_var": None,
            "api_key_visible": False,
        }
    provider = MODEL_PROVIDER_RECIPES.get(s.model_provider)
    env_var = s.model_provider_api_key_env or (provider.api_key_env_var if provider else None)
    key_visible = bool(env_var and os.getenv(env_var))
    if s.model_provider == "google" and not key_visible and os.getenv("GOOGLE_API_KEY"):
        key_visible = True
        env_var = "GOOGLE_API_KEY"
    if not env_var:
        status = "missing_api_key"
        reason = "No API key environment variable is configured."
    elif not key_visible:
        status = "missing_api_key"
        reason = f"{env_var} is not visible."
    else:
        status = s.mentor_model_provider_readiness_status or "untested"
        reason = s.mentor_model_provider_readiness_reason
        if status in {None, "missing_api_key"}:
            status = "untested"
    return {
        "status": status,
        "ready": status == "ready",
        "reason": reason,
        "provider": s.model_provider,
        "provider_display": provider.display_name if provider else s.model_provider,
        "model": s.model_provider_model or (provider.default_model if provider else None),
        "api_key_env_var": env_var,
        "api_key_visible": key_visible,
    }


def _configured_apprentice_command(settings: Settings) -> str | None:
    if settings.worker_agent == "custom":
        template = settings.custom_worker_command_template or ""
        if template:
            try:
                parts = shlex.split(template)
            except ValueError:
                parts = template.split()
            return parts[0] if parts else settings.worker_agent_command
        return settings.worker_agent_command
    recipe = WORKER_AGENT_RECIPES.get(settings.worker_agent)
    return settings.worker_agent_command or (recipe.command_name if recipe else settings.worker_agent)


def apprentice_agent_display_name(settings: Settings | None = None) -> str:
    s = settings or get_settings()
    recipe = WORKER_AGENT_RECIPES.get(s.worker_agent)
    if s.worker_agent == "custom" and s.custom_worker_display_name:
        return s.custom_worker_display_name
    return recipe.display_name if recipe else s.worker_agent


def apprentice_agent_readiness_status(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    command = _configured_apprentice_command(s)
    if s.worker_agent == "custom":
        template = s.custom_worker_command_template or ""
        missing = [token for token in ("{workspace}", "{prompt_file}") if token not in template]
        if missing:
            return {
                "status": "not_ready",
                "ready": False,
                "reason": "Custom Apprentice Agent command template is missing placeholders: " + ", ".join(missing),
                "command": command,
                "command_found": False,
            }
    if not command:
        return {
            "status": "not_ready",
            "ready": False,
            "reason": "Apprentice Agent command is not configured.",
            "command": None,
            "command_found": False,
        }
    _, resolved = resolve_agent_command(s.worker_agent, command)
    if not resolved:
        hint = gui_app_hint(s.worker_agent or "")
        reason = f"Apprentice Agent command not found: {command}"
        if hint:
            reason = f"{reason}. {hint} Install or expose the headless CLI on PATH."
        return {
            "status": "missing_command",
            "ready": False,
            "reason": reason,
            "command": command,
            "command_found": False,
            "command_resolved": None,
        }
    if s.apprentice_agent_readiness_status == "ready":
        return {
            "status": "ready",
            "ready": True,
            "reason": s.apprentice_agent_readiness_reason,
            "command": command,
            "command_found": True,
            "command_resolved": resolved,
        }
    if s.apprentice_agent_readiness_status in {"auth_error", "quota_or_credit_error", "timeout", "failed", "not_ready"}:
        return {
            "status": s.apprentice_agent_readiness_status,
            "ready": False,
            "reason": s.apprentice_agent_readiness_reason,
            "command": command,
            "command_found": True,
            "command_resolved": resolved,
        }
    return {
        "status": "untested",
        "ready": False,
        "reason": "Command was found, but no readiness check has been recorded.",
        "command": command,
        "command_found": True,
        "command_resolved": resolved,
    }


def _readiness_line(label: str, status: str | None, reason: str | None = None) -> str:
    pretty = (status or "not_ready").replace("_", " ").title()
    if status == "ready":
        pretty = "Ready"
    elif status == "not_ready":
        pretty = "Not ready"
    elif status == "missing_api_key":
        pretty = "Not ready"
    elif status == "missing_command":
        pretty = "Not ready"
    elif status == "untested":
        pretty = "Untested"
    suffix = f" - {reason}" if reason else ""
    return f"{label}: {pretty}{suffix}"


def public_settings(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    agent_status = apprentice_agent_readiness_status(s)
    provider_status = mentor_model_provider_readiness(s)
    return {
        "apprentice_agent": apprentice_agent_display_name(s),
        "apprentice_agent_status": agent_status["status"],
        "apprentice_agent_status_reason": agent_status.get("reason"),
        "mentor_model_provider": provider_status["provider_display"] if provider_status.get("provider") else None,
        "mentor_model_provider_status": provider_status["status"],
        "mentor_model_provider_status_reason": provider_status.get("reason"),
        "mentor_model_provider_api_key_env": provider_status.get("api_key_env_var"),
        "mentor_model_provider_api_key_visible": provider_status.get("api_key_visible"),
        "apprenticeship_mode": s.apprenticeship_mode,
        "sensitive_info_masking": s.sensitive_info_masking,
        "max_improvement_loops": s.max_improvement_loops,
        "ecosystem_repo": s.ecosystem_repo,
        "ecosystem_repo_path": str(s.ecosystem_repo_path) if s.ecosystem_repo_path else None,
        "training_contribution_mode": s.training_contribution_mode,
        "ecosystem_storage_backend": s.ecosystem_storage_backend,
        "r2_api_url": s.r2_api_url,
        "r2_bucket": s.r2_bucket,
        "r2_index_prefix": s.r2_index_prefix,
        "r2_contribution_prefix": s.r2_contribution_prefix,
        "r2_seed_prefix": s.r2_seed_prefix,
        "app_home": str(s.app_home),
    }


def public_settings_text(settings: Settings | None = None) -> str:
    s = settings or get_settings()
    data = public_settings(s)
    provider = data["mentor_model_provider"] or "Not configured"
    lines = [
        "Agent Apprenticeship Settings",
        "",
        f"Apprentice Agent: {data['apprentice_agent']}",
        _readiness_line("Apprentice Agent Status", data["apprentice_agent_status"], data.get("apprentice_agent_status_reason")),
        f"Mentor Model Provider: {provider}",
        _readiness_line("Mentor Model Provider Status", data["mentor_model_provider_status"], data.get("mentor_model_provider_status_reason")),
        f"Apprenticeship Mode: {apprenticeship_mode_display(data['apprenticeship_mode'])}",
        f"Sensitive Info Masking: {data['sensitive_info_masking']}",
        f"Maximum Improvement Loops: {data['max_improvement_loops']}",
        f"Public Ecosystem Repo: {data['ecosystem_repo'] or DEFAULT_PUBLIC_ECOSYSTEM_REPO}",
        *([f"Public Ecosystem Repo Path: {data['ecosystem_repo_path']}"] if data.get("ecosystem_repo_path") else []),
        f"Storage backend: {data['ecosystem_storage_backend']}",
        f"R2 ingestion API: {data['r2_api_url'] or 'not configured'}",
        f"R2 bucket: {data['r2_bucket']}",
        f"R2 index prefix: {data['r2_index_prefix']}",
        f"R2 contribution prefix: {data['r2_contribution_prefix']}",
        f"R2 seed prefix: {data['r2_seed_prefix']}",
        f"Agent Training Contribution Mode: {training_contribution_mode_display(data['training_contribution_mode'])}",
        f"App Home: {data['app_home']}",
    ]
    if s.apprenticeship_mode == "autonomous" and not configured_model_provider_ready(s):
        lines.extend(
            [
                "",
                "Autonomous Apprenticeship needs a ready Mentor Model Provider.",
                "Run: apprentice configure model",
            ]
        )
    return "\n".join(lines)


def debug_settings(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    data = s.model_dump(mode="json")
    data["model_provider_ready"] = configured_model_provider_ready(s)
    data["mentor_model_provider_readiness"] = mentor_model_provider_readiness(s)
    data["apprentice_agent_readiness"] = apprentice_agent_readiness_status(s)
    data.pop("openai_api_key", None)
    return data
