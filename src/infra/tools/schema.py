"""Shared schema definitions for the tool layer.

This module centralises metadata contracts so each tool can advertise its
capabilities, declare which canvas stages it supports, and log executions in a
structured way.  Having an explicit schema is the first step toward the
"agents-as-tools" innovation: it gives the orchestrator enough context to pick
the right specialist, compare alternatives, and persist what happened for
memory-aware workflows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """High-level grouping so the router can compose specialists."""

    WORKFLOW = "workflow"
    GEOMETRY = "geometry"
    SHORTCUT = "shortcut"
    RENDERING = "rendering"
    MEMORY = "memory"
    SEARCH = "search"
    OTHER = "other"


class CanvasStage(str, Enum):
    """Stages the main canvas agent moves through when solving a task."""

    CHAT = "chat"
    DRAW = "draw"
    EDIT = "edit"
    DESCRIBE = "describe"
    PICK_PATH = "pick_path"
    GENERATE_IMAGE = "generate_image"


class ToolMetadata(BaseModel):
    """Metadata that accompanies a vanilla or MCP tool."""

    name: str = Field(..., description="Unique identifier for the tool")
    summary: str = Field(..., description="Crisp description surfaced to the LLM")
    category: ToolCategory = Field(
        default=ToolCategory.OTHER, description="Functional cluster for routing"
    )
    stages: List[CanvasStage] = Field(
        default_factory=list,
        description="Canvas stages where this tool excels",
    )
    input_schema: Optional[str] = Field(
        default=None,
        description="Human-friendly description of the expected arguments",
    )
    output_schema: Optional[str] = Field(
        default=None, description="What the tool returns and how it should be used"
    )
    tags: List[str] = Field(default_factory=list, description="Free-form routing hints")
    experimental: bool = Field(
        default=False, description="Flag for partially validated or MVP tools"
    )
    version: str = Field(default="0.1", description="Schema version for migrations")


class ToolExecutionEvent(BaseModel):
    """Observability record emitted whenever a tool is invoked."""

    tool_name: str = Field(..., description="Name matching ToolMetadata.name")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Raw arguments passed to the tool"
    )
    result_preview: Optional[str] = Field(
        default=None,
        description="Short preview logged for memory (e.g., SVG snippet, hash)",
    )
    error: Optional[str] = Field(
        default=None, description="Captured exception message, if any"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when run started",
    )
    finished_at: Optional[datetime] = Field(
        default=None, description="UTC timestamp when run finished"
    )

    @property
    def duration_ms(self) -> Optional[float]:
        """Derived helper so dashboards can display latency without extra math."""

        if not self.finished_at:
            return None
        delta = self.finished_at - self.started_at
        return round(delta.total_seconds() * 1000, 3)


ToolTelemetry = List[ToolExecutionEvent]
