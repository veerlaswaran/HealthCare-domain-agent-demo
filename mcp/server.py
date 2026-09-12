"""
Task 14 — MCP server using fastmcp.

Wraps check_appointment_status as a proper MCP tool via the @mcp.tool()
decorator.  fastmcp mounts its transport at /mcp by default.

MCP endpoint:  POST http://127.0.0.1:8001/mcp

Run:
  python mcp/server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP
from app.tools import check_appointment_status as _check

mcp = FastMCP(name="PractoMCP")


@mcp.tool()
def check_appointment_status(record_id: str) -> dict:
    """
    Look up a Practo clinic appointment by its record ID and return the
    appointment's current status, consultation fee (INR), follow-up
    requirement, and a computed escalation_score in [0, 1].

    The escalation_score combines follow_up_required (60% weight) with a
    recency signal derived from days_since_created (40% weight).
    An escalate flag is set True when escalation_score >= 0.60, indicating
    the case should be reviewed by the patient-experience lead.
    Returns found=False with an error message if the record_id is unknown.

    Args:
        record_id: Appointment record identifier in the format APT-XXXX
                   (e.g. 'APT-0042'). Case-insensitive.
    """
    return _check(record_id)


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8001)
