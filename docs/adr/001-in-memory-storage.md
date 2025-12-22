# ADR 001: In-Memory Storage

## Status
Accepted

## Context
We need to store parking space configurations, current status, and recent images. Options considered:
1. PostgreSQL/MySQL database
2. SQLite file
3. Redis
4. In-memory Python data structures

## Decision
Use in-memory Python data structures (dictionaries, lists).

## Rationale
- **Simplicity**: No database setup, connections, or migrations
- **Cost**: No database service needed (render.io free tier has no database)
- **Performance**: Fastest possible access, no I/O
- **Use Case Fit**: We only care about current state, not historical data
- **Acceptable Loss**: Losing data on restart is acceptable - user just recalibrates

## Consequences

### Positive
- Zero operational complexity
- No database costs
- Instant read/write
- Single container deployment

### Negative
- Data lost on container restart
- No persistence across deployments
- Cannot scale horizontally (single instance only)
- No query capabilities

## Data Structure

```python
state = {
    "admin_password": "64-char-random-string",
    "session_token": None,  # Set when admin authenticates
    "spaces": [
        {
            "id": 1,
            "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            "status": "free",  # or "occupied"
            "confidence": 0.95,
            "last_changed": "2024-01-15T10:30:00Z"
        }
    ],
    "last_update": "2024-01-15T10:30:00Z",
    "recent_images": [
        {"image": "base64...", "timestamp": "...", "detections": [...]}
    ]  # Max 5, FIFO
}
```
