"""Interactive terminal client for the Practo Domain Support Agent API."""

from __future__ import annotations

import os
import sys
import uuid

import httpx


API_URL = os.getenv("PRACTO_API_URL", "http://127.0.0.1:8000").rstrip("/")


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /reset  Start a fresh conversation session\n"
        "  /help   Show this message\n"
        "  /quit   Exit the chatbot\n\n"
        "Try: What is the cancellation fee?\n"
        "     What is the status of appointment APT-0008?\n"
    )


def chat() -> int:
    session_id = str(uuid.uuid4())
    print("=" * 60)
    print("Practo Domain Support Agent")
    print(f"Connected to: {API_URL}")
    print("Type /help for commands. Type /quit to exit.")
    print("=" * 60)

    try:
        with httpx.Client(base_url=API_URL, timeout=15.0) as client:
            health = client.get("/health")
            health.raise_for_status()
            print("API status: ready\n")

            while True:
                try:
                    query = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    return 0

                if not query:
                    continue
                if query.lower() in {"/quit", "/exit", "quit", "exit"}:
                    print("Goodbye!")
                    return 0
                if query.lower() == "/help":
                    print_help()
                    continue
                if query.lower() == "/reset":
                    session_id = str(uuid.uuid4())
                    print("Started a new session.\n")
                    continue

                try:
                    response = client.post(
                        "/ask", json={"query": query, "session_id": session_id}
                    )
                    response.raise_for_status()
                    data = response.json()
                except httpx.HTTPError as exc:
                    print(f"\nCould not reach the API: {exc}\n")
                    continue

                print(f"\nAgent ({data['route']}, turn {data['turn']}):")
                print(data["answer"])
                events = data.get("guardrail_events", [])
                if events:
                    print("Guardrails: " + ", ".join(event["event"] for event in events))
                print()
    except httpx.HTTPError as exc:
        print(f"Cannot start chat: API at {API_URL} is unavailable ({exc}).")
        return 1


if __name__ == "__main__":
    sys.exit(chat())
