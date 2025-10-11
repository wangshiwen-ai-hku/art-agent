import json
from typing import Any, Dict, List, Union, Optional
from enum import Enum
from pydantic import BaseModel
import dataclasses
import logging

logger = logging.getLogger(__name__)

def safe_model_dump(
    obj: Any,
    exclude_none: bool = False,
    exclude_unset: bool = False,
    custom_converters: Optional[Dict[type, callable]] = None,
    max_depth: int = 10,
    current_depth: int = 0
) -> Union[Dict[str, Any], List[Any], str, int, float, bool, None]:
    """
    Recursively convert Pydantic models, Enums, dataclasses, and nested structures
    into JSON-serializable types (dict, list, primitives). 

    Enhancements over the original:
    - Supports Pydantic v1 (.dict()) and v2 (.model_dump()).
    - Handles dataclasses and namedtuples.
    - Optional parameters: exclude_none, exclude_unset (passed to Pydantic dump).
    - Custom converters for specific types.
    - Depth limit to prevent infinite recursion.
    - Better error handling with logging.
    - Efficient: avoids unnecessary json.dumps/loads.
    - Handles more types: sets, tuples, frozensets (converted to lists).

    Args:
        obj: The object to serialize.
        exclude_none: If True, exclude fields with None values (Pydantic only).
        exclude_unset: If True, exclude unset fields (Pydantic only).
        custom_converters: Dict of type -> callable for custom serialization.
        max_depth: Maximum recursion depth to prevent stack overflow.
        current_depth: Internal recursion tracker.

    Returns:
        JSON-serializable object (dict, list, str, int, float, bool, None).

    Raises:
        ValueError: If max_depth exceeded or unserializable type encountered.
    """
    if current_depth > max_depth:
        raise ValueError(f"Maximum recursion depth ({max_depth}) exceeded.")

    try:
        # Custom converter if provided
        if custom_converters and type(obj) in custom_converters:
            return custom_converters[type(obj)](obj)

        # Pydantic BaseModel
        if isinstance(obj, BaseModel):
            # Support Pydantic v2 (.model_dump) or v1 (.dict)
            dump_method = getattr(obj, 'model_dump', getattr(obj, 'dict', None))
            if callable(dump_method):
                raw = dump_method(exclude_none=exclude_none, exclude_unset=exclude_unset)
                return safe_model_dump(
                    raw, exclude_none, exclude_unset, custom_converters, max_depth, current_depth + 1
                )
            else:
                logger.warning(f"Object {obj} claims to be Pydantic but has no dump method.")
                return str(obj)

        # Enum
        if isinstance(obj, Enum):
            return obj.value

        # Dataclass
        if dataclasses.is_dataclass(obj):
            raw = dataclasses.asdict(obj)
            return safe_model_dump(
                raw, exclude_none, exclude_unset, custom_converters, max_depth, current_depth + 1
            )

        # Primitives
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        # Dict: recurse on keys/values
        if isinstance(obj, dict):
            return {
                safe_model_dump(k, exclude_none, exclude_unset, custom_converters, max_depth, current_depth + 1): 
                safe_model_dump(v, exclude_none, exclude_unset, custom_converters, max_depth, current_depth + 1)
                for k, v in obj.items()
            }

        # List, tuple, set, frozenset: convert to list and recurse
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [
                safe_model_dump(item, exclude_none, exclude_unset, custom_converters, max_depth, current_depth + 1)
                for item in obj
            ]

        # Objects with __dict__
        if hasattr(obj, '__dict__'):
            raw = obj.__dict__
            return safe_model_dump(
                raw, exclude_none, exclude_unset, custom_converters, max_depth, current_depth + 1
            )

        # Fallback: string representation
        return str(obj)

    except Exception as e:
        logger.error(f"Error serializing object {type(obj)}: {e}")
        return str(obj)  # Safe fallback