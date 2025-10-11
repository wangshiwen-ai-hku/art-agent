"""Compatibility layer re-exporting the service canvas agent implementation."""

from src.service.canvas_agent import CanvasAgent, CanvasState, ThreadConfiguration

__all__ = ["CanvasAgent", "CanvasState", "ThreadConfiguration"]
