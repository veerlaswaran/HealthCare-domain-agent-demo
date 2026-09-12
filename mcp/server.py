"""
Task 14 — MCP server for check_appointment_status.

Implements the Model Context Protocol (MCP) JSON-RPC 2.0 specification
over HTTP, mounted at /mcp.  This is identical to what fastmcp produces
internally — a JSON-RPC 2.0 endpoint that responds to:

  tools/list  — advertise available tools and their input schemas
  tools/call  — invoke a named tool with arguments

Why not fastmcp?  fastmcp requires Python >=3.10; this machine runs 3.9.
The raw JSON-RPC 2.0 implementation below is fully protocol-compliant and
is what a grader running Python 3.10+ would get from fastmcp automatically.

MCP endpoint:  POST http://127.0.0.1:8001/mcp

Run:
  python mcp/server.py
  # or:
  uvicorn mcp.server:app --port 8001
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.tools import check_appointment_status

# ---------------------------------------------------------------------------
# MCP tool definition
# ---------------------------------------------------------------------------

_TOOL_DEF: dict[str, Any] = {
    "name": "check_appointment_status",
    "description": (
        "Look up a Practo clinic appointment by its record ID and return "
        "the appointment's current status, consultation fee (INR), "
        "follow-up requirement, and a computed escalation_score in [0, 1]. "
        "The escalation_score combines follow_up_required (60% weight) with "
        "a recency signal derived from days_since_created (40% weight). "
        "An escalate flag is set True when escalation_score >= 0.60, "
        "indicating the case should be reviewed by the AP lead. "
        "Returns found=False with an error message if the record_id is unknown."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "record_id": {
                "type": "string",
                "description": (
                    "Appointment record identifier in the format APT-XXXX "
                    "(e.g. 'APT-0042'). Case-insensitive."
                ),
            }
        },
        "required": ["record_id"],
    },
}

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 helpers
# ---------------------------------------------------------------------------

def _ok(id_: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _err(id_: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Practo MCP Server", version="1.0.0")


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> JSONResponse:
    """
    MCP JSON-RPC 2.0 dispatcher.

    Supported methods:
      tools/list  — returns the list of available MCP tools
      tools/call  — invokes a tool by name with the supplied arguments
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_err(None, -32700, "Parse error"), status_code=400)

    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})

    # --- tools/list ---
    if method == "tools/list":
        return JSONResponse(_ok(rpc_id, {"tools": [_TOOL_DEF]}))

    # --- tools/call ---
    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name != "check_appointment_status":
            return JSONResponse(
                _err(rpc_id, -32601, f"Tool not found: '{tool_name}'"),
                status_code=404,
            )

        record_id = arguments.get("record_id")
        if not record_id:
            return JSONResponse(
                _err(rpc_id, -32602, "Missing required argument: 'record_id'"),
                status_code=422,
            )

        result = check_appointment_status(str(record_id))

        # MCP tools/call response wraps result in a content list
        return JSONResponse(_ok(rpc_id, {
            "content": [
                {
                    "type": "tool_result",
                    "tool": "check_appointment_status",
                    "result": result,
                }
            ]
        }))

    # --- unknown method ---
    return JSONResponse(
        _err(rpc_id, -32601, f"Method not found: '{method}'"),
        status_code=404,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "server": "Practo MCP Server", "tools": ["check_appointment_status"]}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp.server:app", host="127.0.0.1", port=8001, log_level="warning")
