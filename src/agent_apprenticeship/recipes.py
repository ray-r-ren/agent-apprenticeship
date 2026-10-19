from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RunnerRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    display_name: str
    command_name: str
    prompt_mode: str
    workspace_mode: str
    writes_mode: str
    stdout_stderr_capture: str = "capture subprocess stdout/stderr into attempt logs"
    success_detection: str = "exit code 0 plus valid agent_trace.json and actual_outputs.json"
    expected_output_contract: list[str] = Field(
        default_factory=lambda: ["agent_trace.json", "actual_outputs.json", "artifacts/"]
    )
    notes: str | None = None


class ModelProviderRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    display_name: str
    api_key_env_var: str
    default_model: str
    endpoint_kind: str
    notes: str | None = None


WORKER_AGENT_RECIPES: dict[str, RunnerRecipe] = {
    "codex": RunnerRecipe(
        agent_id="codex",
        display_name="Codex",
        command_name="codex",
        prompt_mode="codex exec --cd <workspace> --sandbox <sandbox> [--skip-git-repo-check] <prompt>",
        workspace_mode="pass the prepared attempt directory with --cd",
        writes_mode="workspace-write sandbox; write deliverables under ./artifacts/",
        notes="Implemented runner.",
    ),
    "cursor": RunnerRecipe(
        agent_id="cursor",
        display_name="Cursor",
        command_name="cursor-agent",
        prompt_mode="cursor-agent headless mode with --prompt-file, --prompt, -p, or run when supported",
        workspace_mode="pass workspace flags when supported, otherwise run from the prepared attempt directory",
        writes_mode="write deliverables under ./artifacts/",
        notes="Headless adapter with CLI capability detection.",
    ),
    "claude-code": RunnerRecipe(
        agent_id="claude-code",
        display_name="Claude Code",
        command_name="claude",
        prompt_mode="claude -p <prompt> or claude --print <prompt>",
        workspace_mode="run from the prepared attempt directory",
        writes_mode="write deliverables under ./artifacts/",
        notes="Headless adapter with auth/setup failure classification.",
    ),
    "openclaw": RunnerRecipe(
        agent_id="openclaw",
        display_name="OpenClaw",
        command_name="openclaw",
        prompt_mode="openclaw run/exec/session with prompt-file or prompt when supported",
        workspace_mode="pass workspace flags when supported, otherwise run from the prepared attempt directory",
        writes_mode="write deliverables under ./artifacts/",
        notes="Diagnosable headless adapter; reports setup-required or headless-unavailable when needed.",
    ),
    "opencode": RunnerRecipe(
        agent_id="opencode",
        display_name="OpenCode",
        command_name="opencode",
        prompt_mode="opencode run <prompt> or opencode run --prompt-file <prompt_file>",
        workspace_mode="pass workspace flags when supported, otherwise run from the prepared attempt directory",
        writes_mode="write deliverables under ./artifacts/",
        notes="Headless adapter with provider setup failure classification.",
    ),
    "hermes-agent": RunnerRecipe(
        agent_id="hermes-agent",
        display_name="Hermes Agent",
        command_name="hermes",
        prompt_mode="hermes run/chat with prompt-file or prompt when supported",
        workspace_mode="pass workspace flags when supported, otherwise run from the prepared attempt directory",
        writes_mode="write deliverables under ./artifacts/",
        notes="Diagnosable headless adapter; reports setup-required or headless-unavailable when needed.",
    ),
    "custom": RunnerRecipe(
        agent_id="custom",
        display_name="Custom",
        command_name="custom-agent",
        prompt_mode="run the configured command template with {workspace} and {prompt_file}",
        workspace_mode="run command from the prepared attempt directory",
        writes_mode="allow writes inside the prepared workspace when configured",
        notes="Generic command-template runner.",
    ),
}


MODEL_PROVIDER_RECIPES: dict[str, ModelProviderRecipe] = {
    "openai": ModelProviderRecipe(provider_id="openai", display_name="OpenAI", api_key_env_var="OPENAI_API_KEY", default_model="gpt-5-mini", endpoint_kind="native"),
    "anthropic": ModelProviderRecipe(provider_id="anthropic", display_name="Anthropic", api_key_env_var="ANTHROPIC_API_KEY", default_model="claude-sonnet-4-6", endpoint_kind="anthropic_messages"),
    "google": ModelProviderRecipe(provider_id="google", display_name="Google Gemini", api_key_env_var="GEMINI_API_KEY", default_model="gemini-2.5-flash", endpoint_kind="gemini_generate_content"),
    "openrouter": ModelProviderRecipe(provider_id="openrouter", display_name="OpenRouter", api_key_env_var="OPENROUTER_API_KEY", default_model="~openai/gpt-latest", endpoint_kind="openai_compatible"),
}


REMOVED_V0_MODEL_PROVIDER_IDS: tuple[str, ...] = ("deepseek",)


PLANNED_WORKER_AGENT_RECIPES: dict[str, RunnerRecipe] = {
    "gemini": RunnerRecipe(agent_id="gemini", display_name="Gemini CLI", command_name="gemini", prompt_mode="planned headless adapter", workspace_mode="planned", writes_mode="planned"),
    "cline": RunnerRecipe(agent_id="cline", display_name="Cline", command_name="cline", prompt_mode="planned headless adapter", workspace_mode="planned", writes_mode="planned"),
}


PLANNED_MODEL_PROVIDER_RECIPES: dict[str, ModelProviderRecipe] = {
    "xai": ModelProviderRecipe(provider_id="xai", display_name="xAI", api_key_env_var="XAI_API_KEY", default_model="grok-3", endpoint_kind="planned"),
    "kimi-moonshot": ModelProviderRecipe(provider_id="kimi-moonshot", display_name="Kimi / Moonshot AI", api_key_env_var="MOONSHOT_API_KEY", default_model="moonshot-v1-auto", endpoint_kind="planned"),
}


def worker_agent_ids() -> list[str]:
    return list(WORKER_AGENT_RECIPES)


def model_provider_ids() -> list[str]:
    return list(MODEL_PROVIDER_RECIPES)
