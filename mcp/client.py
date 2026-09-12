"""
Task 14 — MCP client script (separate process from the LangGraph agent).

Connects to the MCP server at http://127.0.0.1:8001/mcp and:
  1. Calls tools/list to discover available tools.
  2. Calls check_appointment_status for ≥2 different record IDs,
     printing the standardised MCP JSON-RPC response for each.

The server must be running before this script is executed:
  python mcp/server.py        (in a separate terminal)

Then run this client:
  python mcp/client.py

Alternatively, the demo_mcp() function starts the server in-process
for self-contained testing (used by demos/demo_mcp.py).
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

MCP_URL = "http://127.0.0.1:8001/mcp"
TIMEOUT = 10.0  # seconds


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 request helper
# ---------------------------------------------------------------------------

def _rpc(method: str, params: dict[str, Any]) -> dict:
    """Send one JSON-RPC 2.0 request and return the parsed response."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }
    resp = httpx.post(MCP_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# MCP client operations
# ---------------------------------------------------------------------------

def list_tools() -> list[dict]:
    """Call tools/list and return the tools array."""
    response = _rpc("tools/list", {})
    return response.get("result", {}).get("tools", [])


def call_check_appointment_status(record_id: str) -> dict:
    """Call check_appointment_status via MCP and return the full JSON-RPC response."""
    return _rpc(
        "tools/call",
        {
            "name": "check_appointment_status",
            "arguments": {"record_id": record_id},
        },
    )


# ---------------------------------------------------------------------------
# Main — demonstrates ≥2 tool calls with printed MCP responses
# ---------------------------------------------------------------------------

DEMO_RECORD_IDS = [
    "APT-0008",   # follow_up=True, should escalate
    "APT-0001",   # follow_up=False, should not escalate
    "APT-0028",   # follow_up=True, high days_since_created
    "APT-ZZZZ",   # non-existent — demonstrates error path
]


def _print_response(label: str, response: dict) -> None:
    print(f"\n  {label}")
    print("  " + "-" * 60)
    if "error" in response:
        print(f"  ERROR: {response['error']}")
    else:
        result = response.get("result", {})
        content = result.get("content", [{}])[0].get("result", {})
        print(f"  MCP response (id={response.get('id', '')[:8]}...):")
        for k, v in content.items():
            print(f"    {k:<25}: {v}")
    print()


def main() -> None:
    print("=" * 65)
    print("Practo MCP Client — Task 14")
    print(f"Server: {MCP_URL}")
    print("=" * 65)

    # Step 1 — discover tools
    print("\n[Step 1] tools/list")
    tools = list_tools()
    print(f"  Discovered {len(tools)} tool(s):")
    for t in tools:
        print(f"    name       : {t['name']}")
        print(f"    description: {t['description'][:90]}...")
        req = t.get("inputSchema", {}).get("required", [])
        print(f"    required   : {req}")

    # Step 2 — call tool for ≥2 record IDs
    print("\n[Step 2] tools/call — check_appointment_status")
    for rid in DEMO_RECORD_IDS:
        response = call_check_appointment_status(rid)
        _print_response(f"record_id = {rid}", response)

    print("MCP client demo complete.")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        print(
            "\nERROR: Cannot connect to MCP server at", MCP_URL,
            "\nStart the server first:  python mcp/server.py",
        )
        sys.exit(1)
