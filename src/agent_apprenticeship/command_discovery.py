from __future__ import annotations

import os
import shutil
from pathlib import Path


AGENT_COMMAND_CANDIDATES: dict[str, list[str]] = {
    "codex": ["codex"],
    "cursor": ["cursor-agent", "cursor"],
    "claude-code": ["claude"],
    "openclaw": ["openclaw"],
    "opencode": ["opencode"],
    "hermes-agent": ["hermes"],
}


def common_command_dirs() -> list[Path]:
    if os.getenv("AA_DISABLE_LOCAL_ENV") == "1":
        return []
    home = Path.home()
    candidates = [
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
        home / ".local/bin",
        home / ".npm-global/bin",
        home / ".bun/bin",
        home / ".cargo/bin",
        home / ".volta/bin",
        home / ".yarn/bin",
        home / "Library/pnpm",
        home / ".opencode/bin",
        home / ".hermes/hermes-agent",
    ]
    extra = [Path(p) for p in os.getenv("AA_AGENT_COMMAND_DIRS", "").split(os.pathsep) if p]
    seen: set[str] = set()
    dirs: list[Path] = []
    for path in [*extra, *candidates]:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        dirs.append(path.expanduser())
    return dirs


def explicit_command_dirs() -> list[Path]:
    if os.getenv("AA_DISABLE_LOCAL_ENV") == "1":
        return []
    return [Path(p).expanduser() for p in os.getenv("AA_AGENT_COMMAND_DIRS", "").split(os.pathsep) if p]


def resolve_command(command: str | None) -> str | None:
    if not command:
        return None
    expanded = str(Path(command).expanduser()) if any(command.startswith(p) for p in ("~", ".", "/")) else command
    if os.path.sep in expanded:
        path = Path(expanded)
        return str(path) if path.exists() and os.access(path, os.X_OK) else None
    for directory in explicit_command_dirs():
        candidate = directory / expanded
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which(expanded)
    if found:
        return expanded
    for directory in common_command_dirs():
        candidate = directory / expanded
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def resolve_agent_command(agent_id: str, configured_command: str | None = None) -> tuple[str | None, str | None]:
    candidates = [configured_command] if configured_command else []
    candidates.extend(AGENT_COMMAND_CANDIDATES.get(agent_id, []))
    for candidate in candidates:
        if not candidate:
            continue
        resolved = resolve_command(candidate)
        if resolved:
            return candidate, resolved
    return (candidates[0] if candidates else None), None


def gui_app_hint(agent_id: str) -> str | None:
    app_paths = {
        "cursor": Path("/Applications/Cursor.app"),
        "claude-code": Path("/Applications/Claude.app"),
    }
    path = app_paths.get(agent_id)
    if path and path.exists():
        return f"{path} is installed, but the headless CLI command was not found."
    return None
