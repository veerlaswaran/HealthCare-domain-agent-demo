"""
Task 14 — MCP client script (separate process from the LangGraph agent).

Connects to the MCP server at http://127.0.0.1:8001/mcp and:
  1. Lists available tools via tools/list.
  2. Calls check_appointment_status for ≥2 different record IDs,
     printing the standardised MCP JSON-RPC response for each.

The server must be running before this script is executed:
  python mcp/server.py        (in a separate terminal)

Then run this client:
  python mcp/client.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import Client

MCP_URL = "http://127.0.0.1:8001/mcp"

DEMO_RECORD_IDS = [
    "APT-0008",   # follow_up=True  — should escalate
    "APT-0001",   # follow_up=False — should not escalate
    "APT-0028",   # follow_up=True, high days_since_created
    "APT-ZZZZ",   # non-existent   — demonstrates error path
]


async def main() -> None:
    print("=" * 65)
    print("Practo MCP Client — Task 14")
    print(f"Server: {MCP_URL}")
    print("=" * 65)

    async with Client(MCP_URL) as client:

        # Step 1 — discover tools
        print("\n[Step 1] tools/list")
        tools = await client.list_tools()
        print(f"  Discovered {len(tools)} tool(s):")
        for t in tools:
            print(f"    name       : {t.name}")
            print(f"    description: {t.description[:90]}...")

        # Step 2 — call tool for each record ID
        print("\n[Step 2] tools/call — check_appointment_status")
        for rid in DEMO_RECORD_IDS:
            print(f"\n  record_id = {rid}")
            print("  " + "-" * 55)
            result = await client.call_tool(
                "check_appointment_status",
                {"record_id": rid},
            )
            # CallToolResult.data holds the structured return value when
            # the tool returns a dict; fall back to parsing content[0].text
            if result.data is not None:
                data = result.data
            elif result.content:
                raw = result.content[0]
                text = getattr(raw, 'text', None) or str(raw)
                try:
                    data = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    data = {"raw": text}
            else:
                data = {}
            for k, v in data.items():
                print(f"    {k:<25}: {v}")

    print("\nMCP client demo complete.")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
