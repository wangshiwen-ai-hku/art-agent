"""Running MCP Servers from YAML Configuration"""

import subprocess
import time
import signal
import sys
import os
import argparse
from urllib.parse import urlparse

# Add parent directory to path to import config utility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.manager import config

# Mapping from server names to Python module paths
SERVER_MODULES = {"calculator": "src.infra.mcp_server.calculator"}


def parse_http_url(url):
    """Parse HTTP URL into host, port, and path components."""
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 80,
        "path": parsed.path or "/",
    }


def run_servers(server_names=None):
    """Run MCP servers from YAML configuration.

    Args:
        server_names: List of server names to run. If None, runs all available servers.
    """
    # Load server configurations from YAML
    mcp_servers = config.get_all_mcp_servers()

    # Filter servers if specific names provided
    if server_names:
        mcp_servers = {
            name: server_config
            for name, server_config in mcp_servers.items()
            if name in server_names
        }
        if not mcp_servers:
            print(f"❌ No matching servers found for: {', '.join(server_names)}")
            print(
                f"Available servers: {', '.join(config.get_all_mcp_servers().keys())}"
            )
            return

    # Convert YAML configs to the format expected by the script
    servers = []
    for name, server_config in mcp_servers.items():
        # Skip STDIO transport servers
        if server_config.transport == "stdio":
            print(
                f"⚠️  Skipping {name} server (STDIO transport not supported in this runner)"
            )
            continue

        # Get the Python module for this server
        module = SERVER_MODULES.get(name)
        if not module:
            print(f"⚠️  Unknown server: {name}")
            continue

        # Parse HTTP URL
        if server_config.url:
            http_config = parse_http_url(server_config.url)
            servers.append(
                {
                    "name": name.replace("_", " ").title(),
                    "module": module,
                    "server_config": {
                        "transport": server_config.transport,
                        "host": http_config["host"],
                        "port": http_config["port"],
                        "path": http_config["path"],
                    },
                }
            )

    if not servers:
        print("❌ No HTTP servers found in configuration!")
        return

    processes = []

    def cleanup(signum, frame):
        """Clean up processes on exit."""
        print("\n🛑 Stopping all servers...")
        for p in processes:
            p.terminate()
        sys.exit(0)

    # Set up signal handler
    signal.signal(signal.SIGINT, cleanup)

    print("🚀 Starting MCP Servers...\n")

    # Start each server
    for server in servers:
        # Skip disabled servers
        if not server.get("enabled", True):
            continue

        server_cfg = server["server_config"]
        print(f"Starting {server['name']} server:")
        print(f"  Transport: {server_cfg['transport']}")
        print(f"  Host: {server_cfg['host']}")
        print(f"  Port: {server_cfg['port']}")
        print(f"  Path: {server_cfg.get('path', '/')}")

        # Pass configuration as command-line arguments
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            server["module"],
            "--transport",
            server_cfg["transport"],
            "--host",
            server_cfg["host"],
            "--port",
            str(server_cfg["port"]),
        ]

        # Add path if specified
        if "path" in server_cfg:
            cmd.extend(["--path", server_cfg["path"]])

        process = subprocess.Popen(cmd)
        processes.append(process)
        time.sleep(2)  # Give server time to start

    print("\n✅ All servers started!")
    print("Press Ctrl+C to stop all servers\n")

    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MCP servers from YAML configuration",
        epilog="Example: python run_mcp_servers.py calculator web_search",
    )
    parser.add_argument(
        "servers",
        nargs="*",
        help="Specific servers to run (e.g., calculator web_search). If not specified, runs all available servers.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available servers from config.yaml",
    )

    args = parser.parse_args()

    # List servers and exit if requested
    if args.list:
        print("Available MCP servers from config.yaml:")
        for name, server_config in config.get_all_mcp_servers().items():
            print(f"  - {name} ({server_config.transport} transport)")
        sys.exit(0)

    # Run servers
    run_servers(args.servers if args.servers else None)
