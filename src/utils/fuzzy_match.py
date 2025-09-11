"""
Fuzzy matching utilities for string matching and selection.

This module provides utilities for finding the best match from a collection
of strings using various matching strategies including exact match,
case-insensitive match, substring match, and similarity-based matching.
"""

import difflib
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def find_best_match(
    query: str,
    candidates: List[str],
    similarity_threshold: float = 0.6,
    case_sensitive: bool = False,
) -> Optional[str]:
    """Find the best matching string from a list of candidates using fuzzy matching.

    This function employs a multi-stage matching strategy:
    1. Exact match (optionally case-sensitive)
    2. Case-insensitive exact match (if case_sensitive=False)
    3. Substring match (bidirectional)
    4. Similarity-based match using difflib

    Args:
        query: The string to find a match for.
        candidates: List of candidate strings to match against.
        similarity_threshold: Minimum similarity score for fuzzy matching (0.0-1.0).
                            Default is 0.6.
        case_sensitive: Whether to perform case-sensitive matching. Default is False.

    Returns:
        The best matching candidate string, or None if no good match is found.

    Examples:
        >>> candidates = ["base_planner", "coordinator", "research_agent"]
        >>> find_best_match("planner", candidates)
        'base_planner'

        >>> find_best_match("Planner", candidates, case_sensitive=True)
        None

        >>> find_best_match("planer", candidates)  # typo
        'base_planner'
    """
    if not candidates:
        return None

    if not query:
        return None

    # Stage 1: Exact match
    if query in candidates:
        return query

    # Stage 2: Case-insensitive exact match (if not case sensitive)
    if not case_sensitive:
        query_lower = query.lower()
        for candidate in candidates:
            if candidate.lower() == query_lower:
                return candidate

    # Stage 3: Substring match (bidirectional)
    comparison_query = query if case_sensitive else query.lower()
    for candidate in candidates:
        comparison_candidate = candidate if case_sensitive else candidate.lower()
        if (
            comparison_query in comparison_candidate
            or comparison_candidate in comparison_query
        ):
            logger.debug(f"Fuzzy match: '{query}' -> '{candidate}' (substring match)")
            return candidate

    # Stage 4: Similarity-based match using difflib
    comparison_candidates = (
        candidates if case_sensitive else [c.lower() for c in candidates]
    )
    best_matches = difflib.get_close_matches(
        comparison_query, comparison_candidates, n=1, cutoff=similarity_threshold
    )

    if best_matches:
        best_match = best_matches[0]
        # Find the original case version
        if case_sensitive:
            matched_candidate = best_match
        else:
            for candidate in candidates:
                if candidate.lower() == best_match:
                    matched_candidate = candidate
                    break
            else:
                return None

        logger.debug(
            f"Fuzzy match: '{query}' -> '{matched_candidate}' (similarity match)"
        )
        return matched_candidate

    return None


def find_best_match_from_dict(
    query: str,
    candidates_dict: Dict[str, Any],
    similarity_threshold: float = 0.6,
    case_sensitive: bool = False,
) -> Optional[str]:
    """Find the best matching key from a dictionary using fuzzy matching.

    This is a convenience function that wraps find_best_match for dictionary keys.

    Args:
        query: The string to find a match for.
        candidates_dict: Dictionary with string keys to match against.
        similarity_threshold: Minimum similarity score for fuzzy matching (0.0-1.0).
        case_sensitive: Whether to perform case-sensitive matching.

    Returns:
        The best matching key from the dictionary, or None if no good match is found.

    Examples:
        >>> registry = {"base_planner": PlannerAgent(), "coordinator": CoordAgent()}
        >>> find_best_match_from_dict("planner", registry)
        'base_planner'
    """
    return find_best_match(
        query=query,
        candidates=list(candidates_dict.keys()),
        similarity_threshold=similarity_threshold,
        case_sensitive=case_sensitive,
    )


def get_match_suggestions(
    query: str,
    candidates: List[str],
    max_suggestions: int = 3,
    similarity_threshold: float = 0.3,
    case_sensitive: bool = False,
) -> List[str]:
    """Get multiple match suggestions for a query string.

    This function returns multiple potential matches, useful for providing
    suggestions when no exact match is found.

    Args:
        query: The string to find matches for.
        candidates: List of candidate strings to match against.
        max_suggestions: Maximum number of suggestions to return.
        similarity_threshold: Minimum similarity score for inclusion.
        case_sensitive: Whether to perform case-sensitive matching.

    Returns:
        List of suggested candidate strings, ordered by similarity score.

    Examples:
        >>> candidates = ["base_planner", "coordinator", "research_agent", "planner_v2"]
        >>> get_match_suggestions("planer", candidates)
        ['base_planner', 'planner_v2']
    """
    if not candidates or not query:
        return []

    comparison_query = query if case_sensitive else query.lower()
    comparison_candidates = (
        candidates if case_sensitive else [c.lower() for c in candidates]
    )

    # Get similarity-based matches
    matches = difflib.get_close_matches(
        comparison_query,
        comparison_candidates,
        n=max_suggestions,
        cutoff=similarity_threshold,
    )

    # Convert back to original case if needed
    if case_sensitive:
        return matches
    else:
        suggestions = []
        for match in matches:
            for candidate in candidates:
                if candidate.lower() == match:
                    suggestions.append(candidate)
                    break
        return suggestions
