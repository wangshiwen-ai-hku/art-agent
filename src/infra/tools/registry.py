"""Runtime registry for canvas tools and their metadata."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional

from langchain_core.tools import BaseTool, Tool

from .schema import CanvasStage, ToolExecutionEvent, ToolMetadata, ToolTelemetry


class ToolRegistry:
    """Keeps track of vanilla and MCP tools plus rich metadata.

    The registry is designed to support orchestration innovations such as
    metadata-driven routing, shortcut discovery, and execution logging without
    breaking existing callers that expect LangChain `Tool` objects back.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._vanilla: set[str] = set()
        self._mcp: set[str] = set()
        self._telemetry: Dict[str, ToolTelemetry] = defaultdict(list)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_vanilla(
        self, tool: Tool, metadata: Optional[ToolMetadata] = None
    ) -> None:
        """Register a first-party tool."""

        self._register(tool, metadata)
        self._vanilla.add(tool.name)
        self._mcp.discard(tool.name)

    def register_mcp(self, tool: Tool, metadata: Optional[ToolMetadata] = None) -> None:
        """Register a tool sourced from an MCP server."""

        self._register(tool, metadata)
        self._mcp.add(tool.name)
        self._vanilla.discard(tool.name)

    def _register(self, tool: Tool, metadata: Optional[ToolMetadata]) -> None:
        if not isinstance(tool, (Tool, BaseTool)):
            raise TypeError("Expected a LangChain Tool or BaseTool instance")

        self._tools[tool.name] = tool
        if metadata is None:
            metadata = ToolMetadata(
                name=tool.name,
                summary=(tool.description or tool.name),
                stages=[],
            )
        self._metadata[tool.name] = metadata

    def replace_mcp_tools(
        self,
        tools: Iterable[Tool],
        metadata_map: Optional[Dict[str, ToolMetadata]] = None,
    ) -> None:
        """Replace the list of MCP tools with a new set."""

        # Remove old MCP tools from metadata/registry but keep vanilla entries.
        for name in list(self._mcp):
            self._tools.pop(name, None)
            self._metadata.pop(name, None)

        self._mcp.clear()

        for tool in tools:
            meta = metadata_map.get(tool.name) if metadata_map else None
            self.register_mcp(tool, meta)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_tools(self, names: Iterable[str]) -> List[Tool]:
        result: List[Tool] = []
        for name in names:
            tool = self.get_tool(name)
            if tool:
                result.append(tool)
        return result

    def iter_tools(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def is_vanilla(self, name: str) -> bool:
        return name in self._vanilla

    def is_mcp(self, name: str) -> bool:
        return name in self._mcp

    def tool_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def vanilla_names(self) -> List[str]:
        return sorted(self._vanilla)

    def mcp_names(self) -> List[str]:
        return sorted(self._mcp)

    def metadata(self, name: str) -> Optional[ToolMetadata]:
        return self._metadata.get(name)

    def metadata_all(self) -> Dict[str, ToolMetadata]:
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------
    def record_event(self, event: ToolExecutionEvent) -> None:
        self._telemetry[event.tool_name].append(event)

    def telemetry(self, name: str) -> ToolTelemetry:
        return list(self._telemetry.get(name, []))

    def recent_usage(self, name: str, limit: int = 10) -> ToolTelemetry:
        history = self._telemetry.get(name, [])
        if limit <= 0:
            return list(history)
        return list(history[-limit:])


# Shared singleton used across the backend.
registry = ToolRegistry()

__all__ = ["registry", "ToolRegistry", "ToolMetadata", "ToolExecutionEvent", "CanvasStage"]

