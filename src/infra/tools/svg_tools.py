"""SVG-focused tools stitched into the agents-as-tools registry."""

from __future__ import annotations

import base64
import re
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from langchain_core.tools import tool

from .registry import registry
from .schema import ToolExecutionEvent


def _summarise_result(result: Any) -> Optional[str]:
    """Produce a compact, human-readable preview for telemetry."""

    if result is None:
        return None
    if isinstance(result, str):
        return result[:200]
    if isinstance(result, list):
        if not result:
            return "0 items"
        sample = result[0]
        if isinstance(sample, dict):
            keys = ", ".join(sorted(sample.keys()))
            return f"{len(result)} items (keys: {keys})"
        return f"{len(result)} items (sample: {str(sample)[:80]})"
    if isinstance(result, dict):
        keys = ", ".join(sorted(result.keys()))
        return f"dict[{keys}]"
    return str(result)[:200]


def _record_tool_event(
    name: str,
    arguments: Dict[str, Any],
    result: Any = None,
    error: Optional[Exception] = None,
    started_at: Optional[datetime] = None,
) -> None:
    event = ToolExecutionEvent(
        tool_name=name,
        arguments=arguments,
        result_preview=None if error else _summarise_result(result),
        error=str(error) if error else None,
        started_at=started_at or datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    registry.record_event(event)


def _safe_float_tokens(payload: str) -> List[float]:
    return [
        float(token)
        for token in re.findall(r"[-+]?[0-9]*\.?[0-9]+", payload)
        if token
    ]


def _parse_path_commands(d_attr: str) -> List[Dict[str, Any]]:
    commands: List[Dict[str, Any]] = []
    pattern = r"([MLCQTAZ])([^MLCQTAZ]*)"
    matches = re.findall(pattern, d_attr)

    for command, params in matches:
        commands.append({"type": command, "parameters": _safe_float_tokens(params)})
    return commands


@tool
def convert_svg_to_structured_data_description(svg_code: str) -> Dict[str, Any]:
    """Parse raw SVG into a structured description for human critique."""

    started_at = datetime.now(UTC)
    try:
        root = ET.fromstring(svg_code)
        structured_data: Dict[str, Any] = {
            "width": root.get("width", "unknown"),
            "height": root.get("height", "unknown"),
            "viewBox": root.get("viewBox", "unknown"),
            "elements": [],
        }

        for elem in root.iter():
            if elem.tag.endswith(("path", "rect", "circle", "ellipse", "line", "polygon")):
                element_data = {
                    "type": elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag,
                    "attributes": dict(elem.attrib),
                }
                structured_data["elements"].append(element_data)

        description_lines = [
            f"SVG image {structured_data['width']}x{structured_data['height']}",
            "elements:",
        ]
        for idx, element in enumerate(structured_data.get("elements", []), start=1):
            description_lines.append(f"{idx}. {element['type']} element")

        description = "\n".join(description_lines)
        _record_tool_event(
            "convert_svg_to_structured_data_description",
            {"svg_code": svg_code[:120]},
            description,
            started_at=started_at,
        )
        return {"description": description, "structured": structured_data}

    except Exception as exc:  # noqa: BLE001
        _record_tool_event(
            "convert_svg_to_structured_data_description",
            {"svg_code": svg_code[:120]},
            error=exc,
            started_at=started_at,
        )
        return {"error": f"failed_to_parse: {exc}"}


@tool
def convert_svg_to_paths(svg_code: str) -> List[Dict[str, Any]]:
    """Extract path definitions alongside basic styling information."""

    started_at = datetime.now(UTC)
    try:
        root = ET.fromstring(svg_code)
        paths: List[Dict[str, Any]] = []

        for path_elem in root.findall(".//{*}path"):
            d_attr = path_elem.get("d", "")
            commands = _parse_path_commands(d_attr)
            paths.append(
                {
                    "path_data": d_attr,
                    "commands": commands,
                    "style": {
                        "fill": path_elem.get("fill", "none"),
                        "stroke": path_elem.get("stroke", "none"),
                        "stroke_width": path_elem.get("stroke-width", "1"),
                    },
                }
            )

        _record_tool_event(
            "convert_svg_to_paths",
            {"svg_code": svg_code[:120]},
            paths,
            started_at=started_at,
        )
        return paths

    except Exception as exc:  # noqa: BLE001
        _record_tool_event(
            "convert_svg_to_paths",
            {"svg_code": svg_code[:120]},
            error=exc,
            started_at=started_at,
        )
        return [{"error": f"path_parse_failed: {exc}"}]


@tool
def convert_path_data_to_bezier_segments(path_data: str) -> List[Dict[str, Any]]:
    """Convert a path definition into explicit Bezier segments."""

    started_at = datetime.now(UTC)
    try:
        commands = _parse_path_commands(path_data)
        segments: List[Dict[str, Any]] = []
        current_point = (0.0, 0.0)

        for index, cmd in enumerate(commands):
            params = cmd.get("parameters", [])
            segment: Dict[str, Any] = {
                "segment_id": index,
                "type": cmd.get("type"),
                "start_point": current_point,
            }

            if cmd.get("type") == "C" and len(params) >= 6:
                segment.update(
                    {
                        "control_point1": (params[0], params[1]),
                        "control_point2": (params[2], params[3]),
                        "end_point": (params[4], params[5]),
                        "curve_type": "cubic_bezier",
                    }
                )
                current_point = (params[4], params[5])
            elif cmd.get("type") == "Q" and len(params) >= 4:
                segment.update(
                    {
                        "control_point": (params[0], params[1]),
                        "end_point": (params[2], params[3]),
                        "curve_type": "quadratic_bezier",
                    }
                )
                current_point = (params[2], params[3])
            elif cmd.get("type") in {"M", "L"} and len(params) >= 2:
                segment.update(
                    {
                        "end_point": (params[0], params[1]),
                        "curve_type": "linear",
                    }
                )
                current_point = (params[0], params[1])
            elif cmd.get("type") == "Z":
                segment.update({"curve_type": "close"})

            segments.append(segment)

        _record_tool_event(
            "convert_path_data_to_bezier_segments",
            {"path_data": path_data[:120]},
            segments,
            started_at=started_at,
        )
        return segments

    except Exception as exc:  # noqa: BLE001
        _record_tool_event(
            "convert_path_data_to_bezier_segments",
            {"path_data": path_data[:120]},
            error=exc,
            started_at=started_at,
        )
        return [{"error": f"bezier_conversion_failed: {exc}"}]


@tool
def convert_svg_to_png_base64(svg_code: str) -> str:
    """Render inline SVG into a base64-encoded PNG fallback."""

    started_at = datetime.now(UTC)
    try:
        import cairosvg  # type: ignore

        png_data = cairosvg.svg2png(bytestring=svg_code.encode())
        base64_str = base64.b64encode(png_data).decode("utf-8")
        result = f"data:image/png;base64,{base64_str}"
        _record_tool_event(
            "convert_svg_to_png_base64",
            {"svg_code": svg_code[:120]},
            result,
            started_at=started_at,
        )
        return result

    except ImportError as exc:
        fallback = f"data:image/svg+xml;base64,{base64.b64encode(svg_code.encode()).decode()}"
        _record_tool_event(
            "convert_svg_to_png_base64",
            {"svg_code": svg_code[:120]},
            fallback,
            error=exc,
            started_at=started_at,
        )
        return fallback
    except Exception as exc:  # noqa: BLE001
        _record_tool_event(
            "convert_svg_to_png_base64",
            {"svg_code": svg_code[:120]},
            error=exc,
            started_at=started_at,
        )
        raise


PickPathTools = [
    convert_svg_to_paths,
    convert_path_data_to_bezier_segments,
    convert_svg_to_png_base64,
]
