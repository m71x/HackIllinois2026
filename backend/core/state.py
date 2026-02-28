"""
Shared application state for pipeline statistics and SSE broadcasting.
"""

import asyncio

# Track pipeline statistics (in-memory for hackathon)
pipeline_stats = {
    "stories_ingested": 0,
    "narratives_created": 0,
    "narratives_updated": 0,
    "queue_size": 0,
    "errors": 0,
}

# SSE event queue for real-time updates
sse_subscribers: list[asyncio.Queue] = []


def broadcast_event(event_data: dict):
    """Broadcast an event to all SSE subscribers."""
    for queue in sse_subscribers:
        try:
            queue.put_nowait(event_data)
        except asyncio.QueueFull:
            pass  # Drop events if client is slow
