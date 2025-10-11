"""Central entry-point for accessing canvas tools and MCP integrations."""

from __future__ import annotations

import asyncio
import logging
import pprint
from typing import Dict, Iterable, List, Optional

from langchain_core.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config.manager import config

from .design_tools import design_agent_with_tool
from .draw_canvas_tools import draw_agent_with_tool
from .edit_canvas_tools import edit_agent_with_tool
from .image_tools import edit_image_tool, generate_image_tool
from .metaso import retrieve_relevant_image, retrieve_relevant_video
from .registry import CanvasStage, ToolMetadata, ToolRegistry, registry
from .schema import ToolCategory
from .search import tavily_search
from .triggers import handoff_to_other_agent, plan_tool, step_done, think_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vanilla registration helpers
# ---------------------------------------------------------------------------


def _register(tool: Tool, metadata: ToolMetadata) -> None:
    registry.register_vanilla(tool, metadata)


def _register_vanilla_tools() -> None:
    """Populate the registry with first-party tools plus rich metadata."""

    _register(
        tavily_search,
        ToolMetadata(
            name="tavily_search",
            summary="Search the web for references, mood boards, or design facts.",
            category=ToolCategory.SEARCH,
            stages=[CanvasStage.CHAT, CanvasStage.DESCRIBE],
            tags=["research", "inspiration"],
            output_schema="List of relevant links and descriptions",
        ),
    )
    _register(
        retrieve_relevant_image,
        ToolMetadata(
            name="retrieve_relevant_image",
            summary="Fetch inspirational images aligned with the current topic.",
            category=ToolCategory.SEARCH,
            stages=[CanvasStage.CHAT, CanvasStage.DRAW],
            tags=["reference", "image"],
        ),
    )
    _register(
        retrieve_relevant_video,
        ToolMetadata(
            name="retrieve_relevant_video",
            summary="Find motion or animation references for the concept.",
            category=ToolCategory.SEARCH,
            stages=[CanvasStage.CHAT],
            tags=["reference", "video"],
        ),
    )
    _register(
        handoff_to_other_agent,
        ToolMetadata(
            name="handoff_to_other_agent",
            summary="Signal that another specialist should take over the task.",
            category=ToolCategory.WORKFLOW,
            stages=list(CanvasStage),
            tags=["routing", "handoff"],
        ),
    )
    _register(
        think_tool,
        ToolMetadata(
            name="think_tool",
            summary="Reserve space for internal reasoning before acting.",
            category=ToolCategory.WORKFLOW,
            stages=list(CanvasStage),
            tags=["reflection"],
        ),
    )
    _register(
        plan_tool,
        ToolMetadata(
            name="plan_tool",
            summary="Draft or refine a structured plan of canvas actions.",
            category=ToolCategory.WORKFLOW,
            stages=[CanvasStage.CHAT, CanvasStage.DRAW, CanvasStage.EDIT],
            tags=["planning", "decomposition"],
        ),
    )
    _register(
        step_done,
        ToolMetadata(
            name="step_done",
            summary="Mark the current workflow milestone as complete.",
            category=ToolCategory.WORKFLOW,
            stages=list(CanvasStage),
            tags=["progress"],
        ),
    )
    _register(
        generate_image_tool,
        ToolMetadata(
            name="generate_image_tool",
            summary="Produce reference images or renders for the current brief.",
            category=ToolCategory.RENDERING,
            stages=[CanvasStage.GENERATE_IMAGE, CanvasStage.DRAW],
            tags=["image", "render"],
            experimental=True,
        ),
    )
    _register(
        edit_image_tool,
        ToolMetadata(
            name="edit_image_tool",
            summary="Apply targeted edits to an image asset.",
            category=ToolCategory.RENDERING,
            stages=[CanvasStage.EDIT],
            tags=["image", "edit"],
        ),
    )
    _register(
        draw_agent_with_tool,
        ToolMetadata(
            name="draw_agent_with_tool",
            summary="Draft new SVG geometry based on the design plan and critiques.",
            category=ToolCategory.GEOMETRY,
            stages=[CanvasStage.DRAW],
            tags=["svg", "bezier", "sketch"],
        ),
    )
    _register(
        edit_agent_with_tool,
        ToolMetadata(
            name="edit_agent_with_tool",
            summary="Manipulate existing SVG paths while preserving visual intent.",
            category=ToolCategory.GEOMETRY,
            stages=[CanvasStage.EDIT],
            tags=["svg", "editing"],
        ),
    )
    _register(
        design_agent_with_tool,
        ToolMetadata(
            name="design_agent_with_tool",
            summary="Holistic designer capable of combining draw and edit subtasks.",
            category=ToolCategory.WORKFLOW,
            stages=[CanvasStage.DRAW, CanvasStage.EDIT],
            tags=["design", "orchestration"],
        ),
    )


_register_vanilla_tools()


# ---------------------------------------------------------------------------
# MCP integration
# ---------------------------------------------------------------------------


def _build_mcp_metadata(tools: Iterable[Tool]) -> Dict[str, ToolMetadata]:
    """Create placeholder metadata for MCP tools when the server doesn't provide it."""

    metadata: Dict[str, ToolMetadata] = {}
    for tool in tools:
        metadata[tool.name] = ToolMetadata(
            name=tool.name,
            summary=tool.description or tool.name,
            category=ToolCategory.OTHER,
            stages=list(CanvasStage),
            tags=["mcp"],
        )
    return metadata


async def load_all_mcp_tools() -> None:
    """Load enabled MCP tools defined in `config.yaml` and register them."""

    all_mcp_servers = config.get_all_mcp_servers()
    if not all_mcp_servers:
        logger.info("No MCP servers configured")
        return

    mcp_connections = {}
    enabled_tools_by_server: Dict[str, List[str]] = {}

    for server_name, server_config in all_mcp_servers.items():
        connection_config: Dict[str, Optional[str]] = {
            "transport": server_config.transport
        }

        if server_config.transport == "stdio":
            if server_config.command:
                connection_config["command"] = server_config.command
            if server_config.args:
                connection_config["args"] = server_config.args
        elif server_config.transport in {"streamable_http", "http"}:
            if server_config.transport == "http":
                connection_config["transport"] = "streamable_http"
            if server_config.url:
                connection_config["url"] = server_config.url
        elif server_config.transport == "sse":
            if server_config.url:
                connection_config["url"] = server_config.url
        else:
            logger.warning(
                "Unsupported transport '%s' for MCP server '%s'. Skipping.",
                server_config.transport,
                server_name,
            )
            continue

        mcp_connections[server_name] = connection_config
        enabled_tools_by_server[server_name] = server_config.enabled_tools

    logger.info(
        "Loading MCP tools from %d servers: %s",
        len(mcp_connections),
        list(mcp_connections.keys()),
    )
    logger.debug("MCP connection config:\n%s", pprint.pformat(mcp_connections))

    try:
        client = MultiServerMCPClient(connections=mcp_connections)
        all_tools = await client.get_tools()

        allowed_tool_names = {
            name for names in enabled_tools_by_server.values() for name in names
        }

        enabled_tools: List[Tool] = []
        for tool in all_tools:
            if tool.name in allowed_tool_names:
                enabled_tools.append(tool)
            else:
                logger.debug("Skipping MCP tool '%s' (not enabled).", tool.name)

        logger.info(
            "Loaded %d/%d enabled MCP tools", len(enabled_tools), len(all_tools)
        )

        metadata = _build_mcp_metadata(enabled_tools)
        registry.replace_mcp_tools(enabled_tools, metadata)

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load MCP tools: %s", exc)


def lazy_load_mcp_tools(tool_names: Iterable[str]) -> None:
    """Load MCP tools on-demand when a tool is missing locally."""

    missing = [name for name in tool_names if not registry.is_vanilla(name)]
    if not missing:
        return

    logger.info("Attempting to lazy load MCP tools for: %s", missing)
    try:
        asyncio.get_running_loop()
        logger.warning(
            "Cannot lazy load MCP tools inside an active event loop; skipping."
        )
    except RuntimeError:
        try:
            asyncio.run(load_all_mcp_tools())
        except Exception as exc:  # noqa: BLE001
            logger.error("Lazy MCP load failed: %s", exc)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_available_tool_names() -> List[str]:
    return registry.tool_names()


def get_vanilla_tool_names() -> List[str]:
    return registry.vanilla_names()


def get_mcp_tool_names() -> List[str]:
    return registry.mcp_names()


def get_tools(tool_names: Iterable[str]) -> List[Tool]:
    lazy_load_mcp_tools(tool_names)
    tools = registry.get_tools(tool_names)

    missing = [name for name in tool_names if not registry.has_tool(name)]
    if missing:
        logger.warning("Tools not found in registry: %s", missing)

    logger.debug(
        "Tool request: names=%s | resolved=%s",
        list(tool_names),
        [tool.name for tool in tools],
    )
    return tools


def log_tools_info() -> None:
    logger.info("Vanilla tools: %s", get_vanilla_tool_names())
    logger.info("MCP tools: %s", get_mcp_tool_names())
    logger.info("All tools: %s", get_available_tool_names())


__all__ = [
    "CanvasStage",
    "ToolRegistry",
    "ToolMetadata",
    "get_tools",
    "get_available_tool_names",
    "get_vanilla_tool_names",
    "get_mcp_tool_names",
    "log_tools_info",
    "load_all_mcp_tools",
]

