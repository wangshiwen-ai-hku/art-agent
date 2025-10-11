"""Tools Registry - Central registry for all tools (including vanilla and MCP)."""

from typing import Iterable


def get_tools(tool_names: Iterable[str]):
    """Lazily import the manager to avoid circular imports during tests."""

    from .manager import get_tools as _get_tools

    return _get_tools(tool_names)


__all__ = ["get_tools"]
