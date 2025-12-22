# ParkingSpace Monitor

A self-hosted parking space monitoring system. Uses a smartphone camera as the image source, processes images on the server, and displays occupancy status on a simple public webpage.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 DOCKER CONTAINER (render.io)                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 Python/FastAPI Backend                    │   │
│  │                                                          │   │
│  │  Detection:                                              │   │
│  │  - Pixel difference analysis                             │   │
│  │  - TensorFlow vehicle detection                          │   │
│  │                                                          │   │
│  │  In-Memory Storage:                                      │   │
│  │  - Parking space definitions (from calibration)          │   │
│  │  - Current status per space                              │   │
│  │  - Last 5 captured images                                │   │
│  │  - Admin password (generated at startup)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         ▲                                    │
         │ POST /api/upload (every 30s)       │ GET /
         │ (camera image)                     ▼
┌─────────────────┐                  ┌─────────────────┐
│  Provider       │                  │  Viewer         │
│  (iPhone)       │                  │  (any device)   │
│  /admin page    │                  │  / (public)     │
│  - Calibration  │                  │  - SVG sketch   │
│  - Camera feed  │                  │  - Status       │
└─────────────────┘                  └─────────────────┘
```

## Features

- **No Database**: All state stored in memory (resets on restart)
- **Privacy Friendly**: Public page shows only a schematic, no camera images
- **Simple Auth**: 64-char password generated at startup, shown in logs
- **Dual Detection**: Pixel difference + TensorFlow.js for accuracy
- **Docker Ready**: Single container deployment

## Pages

| Route | Access | Description |
|-------|--------|-------------|
| `/` | Public | Viewer - SVG sketch with colored spaces + last update time |
| `/admin` | Password | Calibration, camera feed, last 5 images |

## Quick Start

### Local Development (Docker)

```bash
# Build and run
docker-compose up --build

# View logs for admin password
docker-compose logs | grep "Admin password"
```

Open http://localhost:8000

### Local Development (without Docker)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

### Deploy to Render.io

1. Push to GitHub
2. Create new Web Service on Render
3. Connect your repo
4. Set:
   - **Environment**: Docker
   - **Instance Type**: Free
5. Deploy
6. Check logs for admin password

## Usage

### 1. Get Admin Password

Check container startup logs for:
```
========================================
ADMIN PASSWORD: aB3x9K2m...
========================================
```

### 2. Calibrate Parking Spaces

1. Open `/admin` on your iPhone
2. Enter the admin password
3. Allow camera access
4. Tap corners to draw parking spaces (4 points per space)
5. Click "Save Calibration"

### 3. Start Monitoring

1. On the admin page, click "Start Monitoring"
2. iPhone will capture and upload every 30 seconds
3. Backend processes images and updates status

### 4. View Status

Open `/` from any device to see the current parking status.

## Configuration

Environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `DETECTION_INTERVAL` | 30 | Seconds between captures |
| `PIXEL_THRESHOLD` | 0.15 | Pixel diff sensitivity (0-1) |
| `TF_CONFIDENCE` | 0.5 | TensorFlow detection threshold |

## Project Structure

```
parkingspace/
├── README.md
├── docker-compose.yml      # Local development
├── Dockerfile              # Production build
├── backend/
│   ├── main.py             # FastAPI application
│   ├── detection.py        # Detection logic
│   ├── requirements.txt
│   └── static/             # Frontend files served by FastAPI
│       ├── index.html      # Public viewer
│       ├── admin.html      # Admin/calibration page
│       ├── css/
│       └── js/
└── docs/
    └── adr/                # Architecture Decision Records
        ├── 001-in-memory-storage.md
        ├── 002-detection-strategy.md
        └── 003-authentication.md
```

## Limitations

- **No Persistence**: Data lost on container restart (by design)
- **Single Provider**: Only one camera source supported
- **Free Tier**: Render.io free tier spins down after 15 min inactivity

## License

MIT
