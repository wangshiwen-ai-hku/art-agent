"""Tools Registry - Central registry for all tools (vanilla and MCP).

This module provides a unified interface for managing and accessing both vanilla tools
and MCP (Model Context Protocol) tools. MCP tools are loaded directly from the 
mcp_servers configuration in config.yaml.
"""

import asyncio
import logging
from typing import Dict, List
from langchain_core.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient
import pprint

# Import vanilla tools
from .search import tavily_search
from .metaso import retrieve_relevant_image, retrieve_relevant_video
from .triggers import handoff_to_other_agent, step_done, think_tool, plan_tool
from .ppt_template_select.ppt_template_select import ppt_template_select
from .exp import generate_experiment
from .flashcard import generate_flashcard
from src.config.manager import config
from .image_generate_edit import generate_image_tool, edit_image_tool
from .canvas_tools import draw_line, draw_circle, draw_rectangle, draw_polygon, draw_bezier_curve, draw_arc, draw_text, draw_ellipse, draw_path
from .canvas_advance_tools import draw_sigmoid, draw_function
logger = logging.getLogger(__name__)

# Vanilla tools registry
vanilla_tools_registry: Dict[str, Tool] = {
    # Search tools
    "tavily_search": tavily_search,
    "retrieve_relevant_image": retrieve_relevant_image,
    "retrieve_relevant_video": retrieve_relevant_video,
    # Agent ability triggers
    "handoff_to_other_agent": handoff_to_other_agent,
    "think_tool": think_tool,
    "plan_tool": plan_tool,
    "step_done": step_done,
    # PPT template selection
    "ppt_template_select": ppt_template_select,
    # mm toosl
    "generate_experiment": generate_experiment,
    "generate_flashcard": generate_flashcard,
    # "generate_image_tool": generate_image_tool,
    # "edit_image_tool": edit_image_tool,
    "generate_image_tool": generate_image_tool,
    "edit_image_tool": edit_image_tool,
    "draw_line": draw_line,
    "draw_circle": draw_circle,
    "draw_rectangle": draw_rectangle,
    "draw_polygon": draw_polygon,
    "draw_bezier_curve": draw_bezier_curve,
    "draw_arc": draw_arc,
    "draw_text": draw_text,
    "draw_ellipse": draw_ellipse,
    "draw_path": draw_path,
    # "clear_canvas": clear_canvas,
    # "export_sketch": export_sketch,
    "draw_sigmoid": draw_sigmoid,
    "draw_function": draw_function,
}

# MCP tools registry - populated dynamically
mcp_tools_registry: Dict[str, Tool] = {}

# Combined tools registry - includes both vanilla and MCP tools
tools_registry: Dict[str, Tool] = vanilla_tools_registry.copy()


async def load_all_mcp_tools() -> None:
    """
    Load only enabled MCP tools from configured servers and populate the MCP tools registry.
    This should be called during application initialization.
    """
    # Get all MCP servers directly from config
    all_mcp_servers = config.get_all_mcp_servers()

    if not all_mcp_servers:
        logger.info("No MCP servers configured")
        return

    # Build server configurations for langchain_mcp_adapters format
    mcp_connections = {}
    enabled_tools_by_server = {}

    for server_name, server_config in all_mcp_servers.items():
        connection_config = {"transport": server_config.transport}

        # Add transport-specific configurations
        if server_config.transport == "stdio":
            if server_config.command:
                connection_config["command"] = server_config.command
            if server_config.args:
                connection_config["args"] = server_config.args
        elif server_config.transport in ["streamable_http", "http"]:
            # Convert http to streamable_http for langchain-mcp-adapters compatibility
            if server_config.transport == "http":
                connection_config["transport"] = "streamable_http"
            if server_config.url:
                connection_config["url"] = server_config.url
        elif server_config.transport == "sse":
            if server_config.url:
                connection_config["url"] = server_config.url
        else:
            logger.warning(
                f"Unsupported transport type '{server_config.transport}' for server '{server_name}'. Skipping this server."
            )
            continue

        mcp_connections[server_name] = connection_config
        logger.debug(
            f"Configured server '{server_name}' with transport '{server_config.transport}'"
        )

        # Store enabled tools for this server
        enabled_tools_by_server[server_name] = server_config.enabled_tools
        logger.info(
            f"Server '{server_name}' enabled tools: {server_config.enabled_tools}"
        )

    logger.info(
        f"Loading MCP tools from {len(mcp_connections)} servers: {list(mcp_connections.keys())}"
    )
    logger.debug(
        f"MCP connections:\n{pprint.pformat(mcp_connections, indent=2, width=80)}"
    )

    # Load MCP tools using the client
    try:
        client = MultiServerMCPClient(connections=mcp_connections)
        all_tools = await client.get_tools()
        # Filter tools to only include enabled ones
        enabled_tools = []

        # Create a mapping of tool names to server names for filtering
        all_enabled_tool_names = set()
        for server_name, tool_names in enabled_tools_by_server.items():
            all_enabled_tool_names.update(tool_names)

        # Filter tools based on enabled_tools configuration
        for tool in all_tools:
            if tool.name in all_enabled_tool_names:
                # Find which server this tool belongs to for enhanced logging
                server_source = None
                for server_name, tool_names in enabled_tools_by_server.items():
                    if tool.name in tool_names:
                        server_source = server_name
                        break

                logger.info(
                    f"Loading enabled tool '{tool.name}' from server '{server_source}'"
                )
                enabled_tools.append(tool)
            else:
                logger.debug(f"Skipping disabled tool '{tool.name}'")

        logger.info(
            f"Successfully loaded {len(enabled_tools)} enabled MCP tools out of {len(all_tools)} available tools"
        )

        # Update the registry with only enabled tools
        update_mcp_tools(enabled_tools)

    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}")


def lazy_load_mcp_tools(tool_names: List[str]) -> None:
    """
    Lazy load MCP tools if any of the requested tools are missing from vanilla tools.

    Args:
        tool_names: List of tool names being requested
    """
    # Check if MCP tools need to be loaded
    if len(mcp_tools_registry) == 0 and tool_names:
        missing_tools = [
            name for name in tool_names if name not in vanilla_tools_registry
        ]
        if missing_tools:
            logger.info(f"Loading MCP tools for missing tools: {missing_tools}")
            try:
                # Check if we're in an event loop
                loop = asyncio.get_running_loop()
                logger.warning(
                    "Cannot load MCP tools in running event loop. MCP tools may not be available."
                )
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                try:
                    asyncio.run(load_all_mcp_tools())
                    logger.info("Successfully loaded MCP tools")
                except Exception as e:
                    logger.error(f"Failed to load MCP tools: {e}")


def update_mcp_tools(tools: List[Tool]) -> None:
    """Update the MCP tools registry with new tools."""
    global mcp_tools_registry, tools_registry

    # Clear existing MCP tools
    mcp_tools_registry.clear()

    # Remove old MCP tools from combined registry
    for tool_name in list(tools_registry.keys()):
        if tool_name not in vanilla_tools_registry:
            del tools_registry[tool_name]

    # Add new MCP tools
    for tool in tools:
        mcp_tools_registry[tool.name] = tool
        tools_registry[tool.name] = tool

    logger.info(
        f"Updated MCP tools registry with {len(tools)} tools: "
        f"{list(mcp_tools_registry.keys())}"
    )


def get_available_tool_names() -> List[str]:
    """
    Get list of all available tool names in the registry.

    Returns:
        List of tool names
    """
    return list(tools_registry.keys())


def get_vanilla_tool_names() -> List[str]:
    """
    Get list of vanilla tool names.

    Returns:
        List of vanilla tool names
    """
    return list(vanilla_tools_registry.keys())


def get_mcp_tool_names() -> List[str]:
    """
    Get list of MCP tool names.

    Returns:
        List of MCP tool names
    """
    return list(mcp_tools_registry.keys())


def log_tools_info():
    """Log tools info."""
    logger.info(f"Vanilla tools registry: {get_vanilla_tool_names()}")
    logger.info(f"MCP tools registry: {get_mcp_tool_names()}")
    logger.info(f"Tools registry: {get_available_tool_names()}")


def get_tools_from_registry(tool_names: List[str]) -> List[Tool]:
    """
    Get tools by names from the unified tools registry.

    Args:
        tool_names: List of tool names to retrieve

    Returns:
        List of Tool objects found in registry

    Note:
        This function only returns tools that are currently in the registry.
        Missing tools are logged but don't raise errors.
    """
    tools = []
    missing_tools = []

    for tool_name in tool_names:
        if tool_name in tools_registry:
            tools.append(tools_registry[tool_name])
        else:
            logger.warning(f"Tool '{tool_name}' not found in registry")
            missing_tools.append(tool_name)

    if missing_tools:
        logger.info(
            f"Missing tools: {missing_tools}. These may be MCP tools that require their servers to be running."
        )

    return tools


def get_tools(tool_names: List[str]) -> List[Tool]:
    """
    Get tools by names from the unified tools registry.

    Args:
        tool_names: List of tool names to retrieve

    Returns:
        List of Tool objects

    Raises:
        ValueError: If no tools are found for the requested tool names
    """
    # Lazy load MCP tools if needed
    lazy_load_mcp_tools(tool_names)

    # Get tools from registry
    tools = get_tools_from_registry(tool_names)

    # Log tools info
    log_tools_info()

    # Raise error if no tools were found at all
    if not tools and tool_names:
        logger.error("No tools were loaded successfully")
        raise ValueError(f"No tools found for requested tools: {tool_names}")

    return tools
