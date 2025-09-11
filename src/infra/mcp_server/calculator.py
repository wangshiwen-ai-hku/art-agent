"""Calculator MCP Server - Simple math operations"""

import math
import argparse
from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Calculator")


@mcp.tool
def calculator_add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool
def calculator_solve_quadratic(a: float, b: float, c: float) -> dict:
    """
    Solve quadratic equation ax² + bx + c = 0.
    Returns dict with roots or error message.
    """
    if a == 0:
        return {"error": "Not a quadratic equation (a=0)"}

    discriminant = b**2 - 4 * a * c

    if discriminant > 0:
        root1 = (-b + math.sqrt(discriminant)) / (2 * a)
        root2 = (-b - math.sqrt(discriminant)) / (2 * a)
        return {"roots": [root1, root2], "type": "two real roots"}
    elif discriminant == 0:
        root = -b / (2 * a)
        return {"roots": [root], "type": "one real root"}
    else:
        real_part = -b / (2 * a)
        imaginary_part = math.sqrt(-discriminant) / (2 * a)
        return {
            "roots": [
                f"{real_part} + {imaginary_part}i",
                f"{real_part} - {imaginary_part}i",
            ],
            "type": "two complex roots",
        }


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculator MCP Server")
    parser.add_argument(
        "--transport", default="http", help="Transport type (http, stdio, websocket)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--path", default="/", help="URL path for the server")

    args = parser.parse_args()

    # Run server with configured parameters
    run_kwargs = {"transport": args.transport, "host": args.host, "port": args.port}

    # Add path parameter if supported by transport
    if args.transport in ["http", "streamable_http"] and args.path != "/":
        run_kwargs["path"] = args.path

    print(
        f"Starting Calculator MCP Server with {args.transport} transport on {args.host}:{args.port}{args.path}"
    )
    mcp.run(**run_kwargs)
