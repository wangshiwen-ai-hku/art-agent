from langchain_core.tools import tool
import numexpr

@tool
def calculator(expression: str) -> str:
    """
    Calculates the result of a mathematical expression using a safe evaluator.
    Use this to perform any mathematical calculations needed to determine coordinates, angles, or dimensions for SVG paths.
    The expression can contain numbers, standard operators, and functions from the math module (e.g., 'math.sin', 'math.pi').
    Example: "200 + 150 * math.cos(math.pi / 3)"
    """
    try:
        # Using numexpr for safe evaluation of mathematical expressions.
        # We provide a limited context with the 'math' module.
        local_dict = {"math": __import__("math")}
        result = numexpr.evaluate(expression, global_dict={}, local_dict=local_dict)
        return f"The result of '{expression}' is {result}."
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}" 