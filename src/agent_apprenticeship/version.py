from __future__ import annotations

from importlib import metadata

from . import __version__


def get_package_version() -> str:
    """Return the installed Agent Apprenticeship package version."""
    if __version__:
        return __version__
    try:
        return metadata.version("agent-apprenticeship")
    except metadata.PackageNotFoundError:
        return __version__
