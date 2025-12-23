"""
ParkingSpace Monitor - FastAPI Backend
"""

import os
import json
import secrets
import logging
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from detection import (
    detect_all_spaces, encode_image_base64, decode_base64_image,
    extract_reference_features, generate_privacy_visualization,
    preload_dnn_model, set_dnn_enabled
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PIXEL_THRESHOLD = float(os.getenv('PIXEL_THRESHOLD', '0.15'))
DNN_CONFIDENCE = float(os.getenv('DNN_CONFIDENCE', '0.5'))
DNN_ENABLED = os.getenv('DNN_ENABLED', 'true').lower() in ('true', '1', 'yes')
DNN_PRELOAD = os.getenv('DNN_PRELOAD', 'true').lower() in ('true', '1', 'yes')
MAX_RECENT_IMAGES = 5
CALIBRATION_FILE = Path(__file__).parent / 'calibration.json'


# In-memory state
class AppState:
    def __init__(self):
        self.admin_password: str = secrets.token_urlsafe(48)[:64]
        self.session_token: Optional[str] = None
        self.spaces: List[Dict] = []
        self.reference_features: Optional[Dict] = None  # ORB features for image alignment
        self.last_update: Optional[str] = None
        self.recent_images: List[Dict] = []  # Max 5, FIFO
        self.last_results: List[Dict] = []  # Latest detection results
        self.last_vehicle_boxes: List[Dict] = []  # Latest detected vehicle boxes
        self._load_calibration()

    def _load_calibration(self):
        """Load calibration from file if it exists."""
        if CALIBRATION_FILE.exists():
            try:
                with open(CALIBRATION_FILE, 'r') as f:
                    data = json.load(f)
                self.spaces = data.get('spaces', [])
                self.reference_features = data.get('reference_features')
                self.last_update = data.get('last_update')
                logger.info(f"Loaded calibration: {len(self.spaces)} spaces, features: {bool(self.reference_features)}")

                # Migration: Extract semantic features if not present
                if self.reference_features and 'semantic_features' not in self.reference_features:
                    self._migrate_semantic_features()
            except Exception as e:
                logger.error(f"Failed to load calibration: {e}")

    def _migrate_semantic_features(self):
        """Extract semantic features from reference image if not present."""
        try:
            # Find a reference image from spaces
            for space in self.spaces:
                ref_image = space.get('reference_image')
                if ref_image:
                    from detection import decode_base64_image, extract_semantic_features
                    image = decode_base64_image(ref_image)
                    semantic_features = extract_semantic_features(image)
                    self.reference_features['semantic_features'] = semantic_features
                    self.save_calibration()
                    logger.info(f"Migrated semantic features: {len(semantic_features.get('grass', []))} grass, "
                               f"{len(semantic_features.get('roads', []))} roads")
                    break
        except Exception as e:
            logger.error(f"Failed to migrate semantic features: {e}")

    def save_calibration(self):
        """Save calibration to file."""
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump({
                    'spaces': self.spaces,
                    'reference_features': self.reference_features,
                    'last_update': self.last_update
                }, f)
            logger.info(f"Saved calibration: {len(self.spaces)} spaces, features: {bool(self.reference_features)}")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")

    def reset(self):
        """Reset to initial state (keeps password and session)."""
        self.spaces = []
        self.reference_features = None
        self.last_update = None
        self.recent_images = []
        self.last_results = []
        self.last_vehicle_boxes = []
        # Delete calibration file
        if CALIBRATION_FILE.exists():
            CALIBRATION_FILE.unlink()


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 50)
    logger.info("PARKING SPACE MONITOR")
    logger.info("=" * 50)
    logger.info(f"ADMIN PASSWORD: {state.admin_password}")
    logger.info("=" * 50)
    logger.info(f"DNN_ENABLED: {DNN_ENABLED}, DNN_PRELOAD: {DNN_PRELOAD}")
    logger.info("=" * 50)

    # Configure OpenCV DNN detection
    set_dnn_enabled(DNN_ENABLED)

    # Optionally preload DNN model in background (disabled by default for low-memory environments)
    if DNN_ENABLED and DNN_PRELOAD:
        import threading
        def load_model():
            logger.info("Preloading OpenCV DNN model...")
            success = preload_dnn_model()
            if success:
                logger.info("OpenCV DNN model ready!")
            else:
                logger.warning("OpenCV DNN model failed to load")

        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()
    elif DNN_ENABLED:
        logger.info("DNN enabled but will load on first detection (saves memory)")
    else:
        logger.info("DNN detection disabled - using pixel diff only")

    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="ParkingSpace Monitor",
    lifespan=lifespan
)


# Pydantic models
class AuthRequest(BaseModel):
    password: str


class AuthResponse(BaseModel):
    token: str


class SpaceDefinition(BaseModel):
    id: int
    polygon: List[List[float]]  # [[x, y], ...]
    reference_image: Optional[str] = None  # base64


class CalibrationRequest(BaseModel):
    spaces: List[SpaceDefinition]


class UploadRequest(BaseModel):
    image: str  # base64


class StatusResponse(BaseModel):
    spaces: List[Dict]
    last_update: Optional[str]


# Auth helper
def verify_session(authorization: Optional[str] = Header(None)) -> bool:
    """Verify session token from Authorization header."""
    if not state.session_token:
        return False
    if not authorization:
        return False
    # Expect "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return False
    return parts[1] == state.session_token


def require_auth(authorization: Optional[str] = Header(None)):
    """Raise 401 if not authenticated."""
    if not verify_session(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")


# API Routes
@app.post("/api/auth", response_model=AuthResponse)
async def authenticate(req: AuthRequest):
    """Authenticate with admin password, get session token."""
    if req.password != state.admin_password:
        raise HTTPException(status_code=401, detail="Invalid password")

    # Generate new session token (invalidates old sessions)
    state.session_token = secrets.token_urlsafe(24)
    logger.info("Admin authenticated, new session created")

    return AuthResponse(token=state.session_token)


@app.post("/api/calibration")
async def save_calibration_endpoint(req: CalibrationRequest, authorization: Optional[str] = Header(None)):
    """Save parking space definitions."""
    require_auth(authorization)

    state.spaces = [s.model_dump() for s in req.spaces]
    state.last_update = datetime.utcnow().isoformat() + 'Z'

    # Extract reference features from the first space's reference image for alignment
    features_extracted = False
    for space in state.spaces:
        ref_image = space.get('reference_image')
        if ref_image:
            try:
                image = decode_base64_image(ref_image)
                state.reference_features = extract_reference_features(image)
                features_extracted = True
                logger.info(f"Extracted {len(state.reference_features.get('keypoints', []))} reference features for alignment")
                break  # Only need one reference image for alignment
            except Exception as e:
                logger.error(f"Failed to extract reference features: {e}")

    state.save_calibration()  # Persist to file
    logger.info(f"Calibration saved: {len(state.spaces)} spaces, features: {features_extracted}")

    return {"status": "ok", "spaces": len(state.spaces), "features_extracted": features_extracted}


@app.get("/api/calibration")
async def get_calibration(authorization: Optional[str] = Header(None)):
    """Get current calibration."""
    require_auth(authorization)
    return {"spaces": state.spaces}


@app.post("/api/upload")
async def upload_image(req: UploadRequest, authorization: Optional[str] = Header(None)):
    """Upload image for processing."""
    require_auth(authorization)

    if not state.spaces:
        raise HTTPException(status_code=400, detail="No spaces configured. Run calibration first.")

    try:
        # Run detection with alignment if features available
        results, annotated, global_info = detect_all_spaces(
            req.image,
            state.spaces,
            pixel_threshold=PIXEL_THRESHOLD,
            dnn_confidence=DNN_CONFIDENCE,
            reference_features=state.reference_features
        )

        # Update space statuses
        for result in results:
            for space in state.spaces:
                if space['id'] == result['id']:
                    space['status'] = result['status']
                    space['confidence'] = result['confidence']
                    space['method'] = result['method']
                    space['vehicle_type'] = result.get('vehicle_type')
                    break

        # Update timestamp
        state.last_update = datetime.utcnow().isoformat() + 'Z'

        # Store latest results and vehicle boxes for public endpoint
        state.last_results = results
        state.last_vehicle_boxes = global_info.get('vehicle_boxes', [])

        # Store annotated image in recent
        annotated_b64 = encode_image_base64(annotated)
        state.recent_images.insert(0, {
            'image': annotated_b64,
            'timestamp': state.last_update,
            'results': results,
            'alignment': global_info.get('alignment'),
            'vehicle_boxes': state.last_vehicle_boxes
        })

        # Keep only last 5
        if len(state.recent_images) > MAX_RECENT_IMAGES:
            state.recent_images = state.recent_images[:MAX_RECENT_IMAGES]

        logger.info(f"Image processed: {len(results)} spaces analyzed, {len(state.last_vehicle_boxes)} vehicles detected")

        return {
            "status": "ok",
            "results": results,
            "timestamp": state.last_update,
            "alignment": global_info.get('alignment'),
            "vehicle_boxes": state.last_vehicle_boxes
        }

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """Get current parking status (public endpoint)."""
    # Return simplified status without actual images (privacy-preserving)
    spaces_status = []
    for space in state.spaces:
        spaces_status.append({
            'id': space.get('id'),
            'polygon': space.get('polygon'),
            'status': space.get('status', 'unknown'),
            'confidence': space.get('confidence', 0)
        })

    # Return vehicle boxes (just bounding boxes, not actual images)
    vehicle_boxes = []
    for vbox in state.last_vehicle_boxes:
        vehicle_boxes.append({
            'type': vbox.get('type'),
            'box': vbox.get('box'),  # Normalized coordinates
            'box_pixels': vbox.get('box_pixels')  # Pixel coordinates
        })

    # Get semantic features from reference (roads, grass, buildings, water)
    semantic_features = None
    if state.reference_features and 'semantic_features' in state.reference_features:
        semantic_features = state.reference_features['semantic_features']

    return {
        "spaces": spaces_status,
        "vehicle_boxes": vehicle_boxes,
        "semantic_features": semantic_features,
        "last_update": state.last_update
    }


@app.get("/api/visualization")
async def get_visualization():
    """Get privacy-preserving visualization (public endpoint).

    Returns an image showing parking spaces and vehicle positions
    without the actual camera image.
    """
    if not state.spaces or not state.last_results:
        raise HTTPException(status_code=404, detail="No data available yet")

    # Use a standard size for the visualization (based on reference image size if available)
    viz_width, viz_height = 800, 450  # Default 16:9

    if state.reference_features and 'image_size' in state.reference_features:
        viz_width = state.reference_features['image_size'][0]
        viz_height = state.reference_features['image_size'][1]

    # Generate privacy visualization
    viz = generate_privacy_visualization(
        (viz_height, viz_width, 3),
        state.spaces,
        state.last_results,
        state.last_vehicle_boxes
    )

    viz_b64 = encode_image_base64(viz)

    return {
        "visualization": viz_b64,
        "last_update": state.last_update
    }


@app.get("/api/recent-images")
async def get_recent_images(authorization: Optional[str] = Header(None)):
    """Get last 5 captured images (admin only)."""
    require_auth(authorization)
    return {"images": state.recent_images}


@app.post("/api/reset")
async def reset_state(authorization: Optional[str] = Header(None)):
    """Reset all state except auth."""
    require_auth(authorization)
    state.reset()
    logger.info("State reset")
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "spaces_configured": len(state.spaces),
        "has_session": state.session_token is not None
    }


@app.get("/api/stats")
async def get_stats(authorization: Optional[str] = Header(None)):
    """Get server statistics (admin only)."""
    require_auth(authorization)

    import psutil
    import platform
    from detection import _dnn_net, _dnn_enabled, _tflite_interpreter, _tflite_available, get_model_status

    # Get process memory info
    process = psutil.Process()
    memory_info = process.memory_info()

    # Get system memory info
    system_memory = psutil.virtual_memory()

    # Get detailed model status for debugging
    model_status = get_model_status()

    # Get uptime
    import time
    uptime_seconds = time.time() - process.create_time()

    return {
        "memory": {
            "process_rss_mb": round(memory_info.rss / 1024 / 1024, 1),
            "process_vms_mb": round(memory_info.vms / 1024 / 1024, 1),
            "system_total_mb": round(system_memory.total / 1024 / 1024, 1),
            "system_available_mb": round(system_memory.available / 1024 / 1024, 1),
            "system_percent_used": system_memory.percent
        },
        "ml_models": {
            "mode": "dual",  # Both backends run in parallel
            "opencv_dnn": {
                "name": "OpenCV DNN MobileNet-SSD",
                "loaded": _dnn_net is not None and _dnn_net is not False,
                "enabled": _dnn_enabled,
                "prototxt_exists": model_status.get('opencv_dnn', {}).get('prototxt_exists'),
                "caffemodel_exists": model_status.get('opencv_dnn', {}).get('caffemodel_exists'),
                "caffemodel_size_mb": model_status.get('opencv_dnn', {}).get('caffemodel_size_mb'),
                "download_error": model_status.get('opencv_dnn', {}).get('download_error')
            },
            "tflite": {
                "name": "TFLite SSD MobileNet V1",
                "loaded": _tflite_interpreter is not None and _tflite_interpreter is not False,
                "available": _tflite_available,
                "model_exists": model_status.get('tflite', {}).get('model_exists'),
                "model_size_mb": model_status.get('tflite', {}).get('model_size_mb')
            }
        },
        # Legacy field for backwards compatibility
        "ml_model": {
            "backend": "dual",
            "name": "OpenCV DNN + TFLite (comparison mode)",
            "loaded": (_dnn_net is not None and _dnn_net is not False) or (_tflite_interpreter is not None and _tflite_interpreter is not False),
            "enabled": _dnn_enabled,
            "prototxt_exists": model_status.get('prototxt_exists'),
            "caffemodel_exists": model_status.get('caffemodel_exists'),
            "caffemodel_size_mb": model_status.get('caffemodel_size_mb'),
            "download_error": model_status.get('download_error')
        },
        "detection": {
            "pixel_threshold": PIXEL_THRESHOLD,
            "dnn_confidence": DNN_CONFIDENCE,
            "dnn_preload": DNN_PRELOAD
        },
        "calibration": {
            "spaces_configured": len(state.spaces),
            "has_reference_features": state.reference_features is not None,
            "recent_images_count": len(state.recent_images)
        },
        "system": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "uptime_seconds": round(uptime_seconds),
            "uptime_formatted": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m"
        }
    }


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# HTML routes
@app.get("/", response_class=HTMLResponse)
async def viewer():
    """Serve public viewer page."""
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/admin", response_class=HTMLResponse)
async def admin():
    """Serve admin page."""
    with open("static/admin.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
