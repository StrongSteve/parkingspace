"""
Parking space detection using pixel difference and ML (OpenCV DNN or TFLite).
Includes image alignment for camera movement compensation.
"""

import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import List, Dict, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Detection backend: 'dnn' (OpenCV DNN) or 'tflite' (TensorFlow Lite)
_dnn_backend = os.getenv('DNN_BACKEND', 'dnn').lower()

# OpenCV DNN net (lazy loaded)
_dnn_net = None
_dnn_enabled = True  # Can be disabled for low-memory environments

# TFLite interpreter (lazy loaded)
_tflite_interpreter = None
_tflite_available = False

# ORB feature detector (reusable)
_orb_detector = None

# VOC class names (MobileNet-SSD trained on VOC for OpenCV DNN)
VOC_VEHICLE_CLASSES = {7: 'car', 14: 'motorbike', 6: 'bus'}  # VOC class indices

# COCO class names (for TFLite models trained on COCO)
COCO_CLASSES = {
    0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat'
}
COCO_VEHICLE_CLASSES = {3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck'}

# Use VOC classes for DNN backend, COCO for TFLite
VEHICLE_CLASSES = VOC_VEHICLE_CLASSES if _dnn_backend == 'dnn' else COCO_VEHICLE_CLASSES


def set_dnn_enabled(enabled: bool):
    """Enable or disable DNN detection globally."""
    global _dnn_enabled
    _dnn_enabled = enabled
    logger.info(f"OpenCV DNN detection {'enabled' if enabled else 'disabled'}")


def _get_orb_detector():
    """Get or create ORB feature detector."""
    global _orb_detector
    if _orb_detector is None:
        _orb_detector = cv2.ORB_create(nfeatures=500)
    return _orb_detector


# Track model download status for debugging
_model_download_error = None

def _download_dnn_model():
    """Download OpenCV DNN model files if not present."""
    global _model_download_error
    import urllib.request
    import os

    model_dir = os.path.dirname(os.path.abspath(__file__))
    prototxt_path = os.path.join(model_dir, 'MobileNetSSD_deploy.prototxt')
    caffemodel_path = os.path.join(model_dir, 'MobileNetSSD_deploy.caffemodel')

    # Check if both files exist and have reasonable size
    if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
        # Verify caffemodel is not empty/corrupted (should be ~23MB)
        caffemodel_size = os.path.getsize(caffemodel_path)
        if caffemodel_size > 20_000_000:  # > 20MB
            logger.info(f"Model files exist: prototxt OK, caffemodel {caffemodel_size / 1024 / 1024:.1f}MB")
            return prototxt_path, caffemodel_path
        else:
            logger.warning(f"Caffemodel too small ({caffemodel_size} bytes), re-downloading...")
            os.remove(caffemodel_path)

    # Multiple URLs to try for caffemodel (23MB)
    # PINTO0309 repo is verified working as of Dec 2025
    caffemodel_urls = [
        # PINTO0309 repo (verified working, 23MB)
        "https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel",
        # Alternative: opencv_extra testdata (may have size limits)
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.caffemodel",
    ]

    prototxt_url = "https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.prototxt"

    try:
        if not os.path.exists(prototxt_path):
            logger.info("Downloading MobileNet-SSD prototxt...")
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
            logger.info("Prototxt downloaded successfully")

        if not os.path.exists(caffemodel_path):
            logger.info("Downloading MobileNet-SSD caffemodel (~23MB)...")

            download_success = False
            for i, url in enumerate(caffemodel_urls):
                try:
                    logger.info(f"Trying URL {i+1}/{len(caffemodel_urls)}: {url[:60]}...")
                    urllib.request.urlretrieve(url, caffemodel_path)

                    # Verify download size
                    size = os.path.getsize(caffemodel_path)
                    if size > 20_000_000:
                        logger.info(f"Caffemodel downloaded: {size / 1024 / 1024:.1f}MB")
                        download_success = True
                        break
                    else:
                        logger.warning(f"Downloaded file too small ({size} bytes), trying next URL...")
                        os.remove(caffemodel_path)
                except Exception as url_error:
                    logger.warning(f"URL {i+1} failed: {url_error}")
                    if os.path.exists(caffemodel_path):
                        os.remove(caffemodel_path)
                    continue

            if not download_success:
                _model_download_error = "All download URLs failed"
                logger.error("Failed to download caffemodel from any URL")
                return None, None

        _model_download_error = None
        logger.info("MobileNet-SSD model ready")
        return prototxt_path, caffemodel_path

    except Exception as e:
        _model_download_error = str(e)
        logger.error(f"Failed to download DNN model: {e}")
        # Clean up partial downloads
        for path in [prototxt_path, caffemodel_path]:
            if os.path.exists(path):
                os.remove(path)
        return None, None


def get_model_status() -> dict:
    """Get current model status for debugging."""
    import os
    model_dir = os.path.dirname(os.path.abspath(__file__))

    # OpenCV DNN model files
    prototxt_path = os.path.join(model_dir, 'MobileNetSSD_deploy.prototxt')
    caffemodel_path = os.path.join(model_dir, 'MobileNetSSD_deploy.caffemodel')

    # TFLite model file
    tflite_path = os.path.join(model_dir, 'ssd_mobilenet_v1.tflite')

    return {
        # OpenCV DNN status
        'opencv_dnn': {
            'prototxt_exists': os.path.exists(prototxt_path),
            'caffemodel_exists': os.path.exists(caffemodel_path),
            'caffemodel_size_mb': round(os.path.getsize(caffemodel_path) / 1024 / 1024, 1) if os.path.exists(caffemodel_path) else 0,
            'model_loaded': _dnn_net is not None and _dnn_net is not False,
            'download_error': _model_download_error
        },
        # TFLite status
        'tflite': {
            'model_exists': os.path.exists(tflite_path),
            'model_size_mb': round(os.path.getsize(tflite_path) / 1024 / 1024, 1) if os.path.exists(tflite_path) else 0,
            'interpreter_loaded': _tflite_interpreter is not None and _tflite_interpreter is not False,
            'available': _tflite_available
        },
        # Legacy fields for backwards compatibility
        'prototxt_exists': os.path.exists(prototxt_path),
        'caffemodel_exists': os.path.exists(caffemodel_path),
        'caffemodel_size_mb': round(os.path.getsize(caffemodel_path) / 1024 / 1024, 1) if os.path.exists(caffemodel_path) else 0,
        'download_error': _model_download_error,
        'dnn_net_loaded': _dnn_net is not None and _dnn_net is not False,
        'dnn_enabled': _dnn_enabled,
        'backend': 'dual'  # Now running both
    }


def _load_dnn_model():
    """Lazy load OpenCV DNN model (MobileNet-SSD, ~23MB, low memory)."""
    global _dnn_net, _dnn_enabled

    if not _dnn_enabled:
        return None

    if _dnn_net is None:
        try:
            prototxt_path, caffemodel_path = _download_dnn_model()
            if prototxt_path is None:
                _dnn_net = False
                return None

            logger.info("Loading OpenCV DNN model...")
            _dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            logger.info("OpenCV DNN model loaded successfully (~30MB memory)")
        except Exception as e:
            logger.warning(f"Failed to load DNN model: {e}")
            _dnn_net = False

    return _dnn_net if _dnn_net else None


def preload_dnn_model() -> bool:
    """Preload DNN model at startup.

    Returns True if model loaded successfully, False otherwise.
    """
    if _dnn_backend == 'tflite':
        return _load_tflite_model() is not None
    else:
        return _load_dnn_model() is not None


# ============================================================================
# TFLite Backend (alternative to OpenCV DNN)
# ============================================================================

def _download_tflite_model():
    """Download TFLite model if not present."""
    import urllib.request

    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, 'ssd_mobilenet_v1.tflite')

    if os.path.exists(model_path):
        return model_path

    # SSD MobileNet V1 quantized (~4MB) - good balance of size and accuracy
    model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"

    try:
        import zipfile
        import tempfile

        logger.info("Downloading SSD MobileNet V1 TFLite model (~4MB)...")
        zip_path = os.path.join(tempfile.gettempdir(), 'ssd_mobilenet.zip')
        urllib.request.urlretrieve(model_url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract the .tflite file
            for name in zip_ref.namelist():
                if name.endswith('.tflite'):
                    with zip_ref.open(name) as src, open(model_path, 'wb') as dst:
                        dst.write(src.read())
                    break

        os.remove(zip_path)
        logger.info("TFLite model downloaded successfully")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download TFLite model: {e}")
        return None


def _load_tflite_model():
    """Lazy load TFLite model."""
    global _tflite_interpreter, _tflite_available, _dnn_enabled

    if not _dnn_enabled:
        return None

    if _tflite_interpreter is None:
        try:
            # Try to import tflite_runtime first (lighter), fall back to tensorflow
            try:
                from tflite_runtime.interpreter import Interpreter
                logger.info("Using tflite_runtime")
            except ImportError:
                try:
                    from tensorflow.lite.python.interpreter import Interpreter
                    logger.info("Using tensorflow.lite")
                except ImportError:
                    logger.warning("Neither tflite_runtime nor tensorflow available")
                    _tflite_interpreter = False
                    _tflite_available = False
                    return None

            model_path = _download_tflite_model()
            if model_path is None:
                _tflite_interpreter = False
                return None

            logger.info("Loading TFLite model...")
            _tflite_interpreter = Interpreter(model_path=model_path)
            _tflite_interpreter.allocate_tensors()
            _tflite_available = True
            logger.info("TFLite model loaded successfully (~20MB memory)")
        except Exception as e:
            logger.warning(f"Failed to load TFLite model: {e}")
            _tflite_interpreter = False
            _tflite_available = False

    return _tflite_interpreter if _tflite_interpreter else None


def _detect_with_tflite(image: np.ndarray) -> Tuple[List, List, List]:
    """Run detection using TFLite model.

    Returns: (boxes, classes, scores) - all as numpy arrays
    """
    interpreter = _load_tflite_model()
    if interpreter is None:
        return [], [], []

    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Get input shape
        input_shape = input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]

        # Preprocess image
        img_resized = cv2.resize(image, (width, height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0).astype(np.uint8)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get outputs
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # [ymin, xmin, ymax, xmax]
        classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(int)
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        return boxes, classes, scores
    except Exception as e:
        logger.error(f"TFLite detection error: {e}")
        return [], [], []


def _detect_with_opencv_dnn(image: np.ndarray) -> Tuple[List, List, List]:
    """Run detection using OpenCV DNN model.

    Returns: (boxes, classes, scores) - all as numpy arrays
    """
    net = _load_dnn_model()
    if net is None:
        return [], [], []

    try:
        # OpenCV DNN detection
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Parse detections - shape is [1, 1, N, 7]
        # Each detection: [batch_id, class_id, confidence, x1, y1, x2, y2]
        boxes = []
        classes = []
        scores = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.15:  # Low threshold, filter later
                class_id = int(detections[0, 0, i, 1])
                x1 = detections[0, 0, i, 3]
                y1 = detections[0, 0, i, 4]
                x2 = detections[0, 0, i, 5]
                y2 = detections[0, 0, i, 6]
                # Convert to [ymin, xmin, ymax, xmax] format
                boxes.append([y1, x1, y2, x2])
                classes.append(class_id)
                scores.append(float(confidence))

        boxes = np.array(boxes) if boxes else np.array([])
        classes = np.array(classes) if classes else np.array([])
        scores = np.array(scores) if scores else np.array([])

        return boxes, classes, scores
    except Exception as e:
        logger.error(f"OpenCV DNN detection error: {e}")
        return [], [], []


def extract_semantic_features(image: np.ndarray) -> Dict:
    """Extract semantic features (roads, grass, buildings, water) from reference image.

    Uses color-based segmentation and morphological operations to identify
    different terrain types in aerial/parking lot images.

    Returns dict with SVG path data for each feature type.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    semantic_features = {
        'grass': [],
        'roads': [],
        'buildings': [],
        'water': [],
        'markings': [],  # Road markings (white/yellow lines)
        'image_size': [w, h]
    }

    # Grass detection (green hues)
    # HSV: H=35-85 (green range), S>40 (needs saturation), V>40
    grass_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))

    # Water detection (blue hues) - more restrictive
    # HSV: H=100-125 (blue range), S>50 (needs good saturation)
    water_mask = cv2.inRange(hsv, (100, 50, 40), (125, 255, 255))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))

    # Roads/pavement detection (gray, low saturation, medium brightness)
    # This is the main parking lot surface - light gray asphalt/concrete
    gray_mask = cv2.inRange(hsv, (0, 0, 100), (180, 40, 220))
    # Also include slightly tinted grays
    gray_mask2 = cv2.inRange(hsv, (0, 0, 120), (180, 30, 200))
    gray_mask = cv2.bitwise_or(gray_mask, gray_mask2)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, np.ones((20, 20), np.uint8))

    # Buildings detection - only detect distinct building structures
    # Look for darker, more saturated brownish areas (actual roofs)
    # Much more restrictive - only strong brown/terracotta colors
    building_mask = cv2.inRange(hsv, (5, 80, 60), (20, 255, 180))
    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_OPEN, np.ones((20, 20), np.uint8))

    # Road markings detection (white and yellow lines)
    # White markings: very high brightness, low saturation
    white_marking_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    # Yellow markings: yellow hue, high saturation, high brightness
    yellow_marking_mask = cv2.inRange(hsv, (15, 80, 150), (35, 255, 255))
    markings_mask = cv2.bitwise_or(white_marking_mask, yellow_marking_mask)
    # Clean up - small morphological operations to keep thin lines
    markings_mask = cv2.morphologyEx(markings_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    markings_mask = cv2.morphologyEx(markings_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Priority: grass > roads > buildings > water (grass most visible in parking lot)
    # Remove grass from other masks
    water_mask = cv2.bitwise_and(water_mask, cv2.bitwise_not(grass_mask))
    gray_mask = cv2.bitwise_and(gray_mask, cv2.bitwise_not(grass_mask))
    building_mask = cv2.bitwise_and(building_mask, cv2.bitwise_not(grass_mask))
    building_mask = cv2.bitwise_and(building_mask, cv2.bitwise_not(gray_mask))

    # Convert masks to simplified SVG paths
    def mask_to_paths(mask, min_area=500, max_area_ratio=0.5):
        """Convert binary mask to simplified SVG path data."""
        paths = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = w * h

        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / total_area

            # Skip too small or too large (likely background)
            if area < min_area or area_ratio > max_area_ratio:
                continue

            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)

            if len(simplified) < 3:
                continue

            # Convert to normalized SVG path
            points = simplified.reshape(-1, 2)
            # Normalize to 0-1 range
            normalized = [(float(p[0]) / w, float(p[1]) / h) for p in points]

            # Create SVG path string (with normalized coords, will be scaled in frontend)
            path_d = f"M {normalized[0][0]},{normalized[0][1]}"
            for p in normalized[1:]:
                path_d += f" L {p[0]},{p[1]}"
            path_d += " Z"

            paths.append({'path': path_d, 'area': float(area_ratio)})

        return paths

    # Use higher min_area thresholds to avoid noise
    semantic_features['grass'] = mask_to_paths(grass_mask, min_area=2000, max_area_ratio=0.4)
    semantic_features['water'] = mask_to_paths(water_mask, min_area=1000, max_area_ratio=0.2)
    semantic_features['roads'] = mask_to_paths(gray_mask, min_area=5000, max_area_ratio=0.6)
    semantic_features['buildings'] = mask_to_paths(building_mask, min_area=3000, max_area_ratio=0.15)
    # Markings can be small, so lower threshold
    semantic_features['markings'] = mask_to_paths(markings_mask, min_area=200, max_area_ratio=0.1)

    logger.info(f"Extracted semantic features: {len(semantic_features['grass'])} grass, "
                f"{len(semantic_features['roads'])} roads, {len(semantic_features['buildings'])} buildings, "
                f"{len(semantic_features['water'])} water, {len(semantic_features['markings'])} markings")

    return semantic_features


def extract_reference_features(image: np.ndarray) -> Dict:
    """Extract ORB features from reference image for alignment.

    Returns dict with keypoints and descriptors that can be JSON serialized.
    """
    orb = _get_orb_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if keypoints is None or len(keypoints) == 0:
        return {'keypoints': [], 'descriptors': None}

    # Convert keypoints to serializable format
    kp_data = []
    for kp in keypoints:
        kp_data.append({
            'pt': [float(kp.pt[0]), float(kp.pt[1])],
            'size': float(kp.size),
            'angle': float(kp.angle),
            'response': float(kp.response),
            'octave': int(kp.octave)
        })

    # Convert descriptors to list
    desc_list = descriptors.tolist() if descriptors is not None else None

    # Also extract semantic features
    semantic_features = extract_semantic_features(image)

    return {
        'keypoints': kp_data,
        'descriptors': desc_list,
        'image_size': [image.shape[1], image.shape[0]],
        'semantic_features': semantic_features
    }


def load_reference_features(features_dict: Dict) -> Tuple[List, Optional[np.ndarray]]:
    """Load reference features from serialized dict back to OpenCV format."""
    if not features_dict or not features_dict.get('keypoints'):
        return [], None

    keypoints = []
    for kp_data in features_dict['keypoints']:
        kp = cv2.KeyPoint(
            x=kp_data['pt'][0],
            y=kp_data['pt'][1],
            size=kp_data['size'],
            angle=kp_data['angle'],
            response=kp_data['response'],
            octave=kp_data['octave']
        )
        keypoints.append(kp)

    descriptors = None
    if features_dict.get('descriptors'):
        descriptors = np.array(features_dict['descriptors'], dtype=np.uint8)

    return keypoints, descriptors


def align_image_to_reference(
    current_image: np.ndarray,
    reference_features: Dict,
    debug_info: Dict = None
) -> Tuple[np.ndarray, bool, Dict]:
    """Align current image to reference using feature matching.

    Returns:
        Tuple of (aligned_image, success, alignment_info)
    """
    alignment_info = {
        'aligned': False,
        'matches_found': 0,
        'inliers': 0,
        'transform_type': 'none'
    }

    if not reference_features or not reference_features.get('keypoints'):
        if debug_info is not None:
            debug_info['alignment'] = alignment_info
        return current_image, False, alignment_info

    try:
        orb = _get_orb_detector()

        # Get reference keypoints and descriptors
        ref_keypoints, ref_descriptors = load_reference_features(reference_features)
        ref_size = reference_features.get('image_size', [current_image.shape[1], current_image.shape[0]])

        if ref_descriptors is None or len(ref_keypoints) < 4:
            alignment_info['error'] = 'insufficient_reference_features'
            if debug_info is not None:
                debug_info['alignment'] = alignment_info
            return current_image, False, alignment_info

        # Detect features in current image
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        curr_keypoints, curr_descriptors = orb.detectAndCompute(gray, None)

        if curr_descriptors is None or len(curr_keypoints) < 4:
            alignment_info['error'] = 'insufficient_current_features'
            if debug_info is not None:
                debug_info['alignment'] = alignment_info
            return current_image, False, alignment_info

        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(curr_descriptors, ref_descriptors, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        alignment_info['matches_found'] = len(good_matches)

        if len(good_matches) < 10:
            alignment_info['error'] = 'insufficient_matches'
            if debug_info is not None:
                debug_info['alignment'] = alignment_info
            return current_image, False, alignment_info

        # Extract matched point coordinates
        src_pts = np.float32([curr_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([ref_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            alignment_info['error'] = 'homography_failed'
            if debug_info is not None:
                debug_info['alignment'] = alignment_info
            return current_image, False, alignment_info

        inliers = int(mask.sum()) if mask is not None else 0
        alignment_info['inliers'] = inliers

        if inliers < 8:
            alignment_info['error'] = 'insufficient_inliers'
            if debug_info is not None:
                debug_info['alignment'] = alignment_info
            return current_image, False, alignment_info

        # Warp current image to align with reference
        aligned = cv2.warpPerspective(
            current_image, H,
            (ref_size[0], ref_size[1]),
            flags=cv2.INTER_LINEAR
        )

        alignment_info['aligned'] = True
        alignment_info['transform_type'] = 'homography'
        alignment_info['homography'] = H.tolist()

        if debug_info is not None:
            debug_info['alignment'] = alignment_info

        logger.info(f"Image aligned: {len(good_matches)} matches, {inliers} inliers")
        return aligned, True, alignment_info

    except Exception as e:
        logger.error(f"Image alignment failed: {e}")
        alignment_info['error'] = str(e)
        if debug_info is not None:
            debug_info['alignment'] = alignment_info
        return current_image, False, alignment_info


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image to numpy array (BGR format for OpenCV)."""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image_array = np.array(image.convert('RGB'))
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


def encode_image_base64(image: np.ndarray, quality: int = 70) -> str:
    """Encode numpy array (BGR) to base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"


def extract_region(image: np.ndarray, polygon: List[List[float]], debug_info: dict = None) -> np.ndarray:
    """Extract region defined by polygon from image.

    Args:
        image: Source image (BGR)
        polygon: List of [x, y] coordinates (normalized 0-1)
        debug_info: Optional dict to store debug information

    Returns:
        Cropped and masked region
    """
    h, w = image.shape[:2]

    # Convert normalized coords to pixel coords
    points = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon], np.int32)

    if debug_info is not None:
        debug_info['image_size'] = f"{w}x{h}"
        debug_info['polygon_pixels'] = points.tolist()

    # Get bounding box
    x, y, bw, bh = cv2.boundingRect(points)

    if debug_info is not None:
        debug_info['bounding_box'] = {'x': x, 'y': y, 'width': bw, 'height': bh}

    # Ensure bounds
    x, y = max(0, x), max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)

    if bw <= 0 or bh <= 0:
        if debug_info is not None:
            debug_info['error'] = 'Invalid bounding box dimensions'
        return np.zeros((10, 10, 3), dtype=np.uint8)

    # Crop region
    cropped = image[y:y+bh, x:x+bw].copy()

    if debug_info is not None:
        debug_info['cropped_size'] = f"{bw}x{bh}"

    # Create mask for polygon within cropped region
    mask = np.zeros((bh, bw), dtype=np.uint8)
    shifted_points = points - [x, y]
    cv2.fillPoly(mask, [shifted_points], 255)

    # Apply mask
    masked = cv2.bitwise_and(cropped, cropped, mask=mask)

    if debug_info is not None:
        debug_info['masked_pixels'] = int(np.count_nonzero(mask))

    return masked


def pixel_difference(current: np.ndarray, reference: np.ndarray, debug_info: dict = None) -> Tuple[float, float]:
    """Calculate pixel difference between current and reference images.

    Returns:
        Tuple of (difference_ratio, confidence)
    """
    if debug_info is not None:
        debug_info['current_shape'] = str(current.shape)
        debug_info['reference_shape'] = str(reference.shape)

    if current.shape != reference.shape:
        # Resize reference to match current
        reference = cv2.resize(reference, (current.shape[1], current.shape[0]))
        if debug_info is not None:
            debug_info['reference_resized'] = True

    # Convert to grayscale
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    if debug_info is not None:
        debug_info['current_mean_intensity'] = float(np.mean(current_gray))
        debug_info['reference_mean_intensity'] = float(np.mean(reference_gray))

    # Apply Gaussian blur to reduce noise
    current_blur = cv2.GaussianBlur(current_gray, (5, 5), 0)
    reference_blur = cv2.GaussianBlur(reference_gray, (5, 5), 0)

    # Calculate absolute difference
    diff = cv2.absdiff(current_blur, reference_blur)

    if debug_info is not None:
        debug_info['diff_mean'] = float(np.mean(diff))
        debug_info['diff_max'] = float(np.max(diff))
        debug_info['diff_std'] = float(np.std(diff))

    # Threshold to binary - using adaptive threshold value
    threshold_value = 25  # Lower threshold for better sensitivity
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    if debug_info is not None:
        debug_info['threshold_value'] = threshold_value

    # Count non-zero pixels (only in the masked area)
    # Create mask for non-black pixels in reference (the actual region)
    ref_mask = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) > 0

    non_zero = np.count_nonzero(thresh[ref_mask]) if np.any(ref_mask) else np.count_nonzero(thresh)
    total = np.count_nonzero(ref_mask) if np.any(ref_mask) else thresh.size

    if debug_info is not None:
        debug_info['non_zero_diff_pixels'] = int(non_zero)
        debug_info['total_region_pixels'] = int(total)

    if total == 0:
        return 0.0, 0.0

    diff_ratio = non_zero / total

    if debug_info is not None:
        debug_info['diff_ratio'] = round(diff_ratio, 4)

    # Confidence based on how decisive the result is
    # High confidence if clearly different (>0.2) or clearly same (<0.05)
    if diff_ratio > 0.25:
        confidence = min(0.95, 0.7 + diff_ratio)
    elif diff_ratio < 0.05:
        confidence = min(0.95, 0.9 - diff_ratio * 2)
    else:
        # Ambiguous range - still reasonably confident
        confidence = 0.6 + abs(diff_ratio - 0.1) * 2

    if debug_info is not None:
        debug_info['pixel_confidence'] = round(confidence, 4)

    return diff_ratio, confidence


def boxes_overlap(box1, box2):
    """Check if two boxes overlap. Boxes are [ymin, xmin, ymax, xmax] normalized."""
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    # Check if boxes overlap
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False
    return True


def box_overlap_ratio(box1, box2):
    """Calculate how much of box1 is covered by box2. Returns 0-1."""
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)

    if box1_area == 0:
        return 0.0

    return inter_area / box1_area


def _analyze_detections_for_space(boxes, classes, scores, vehicle_classes, polygon, image_shape):
    """Analyze detection results for a parking space.

    Returns dict with detection analysis.
    """
    h, w = image_shape[:2]

    # Convert polygon to normalized bounding box
    poly_xs = [p[0] for p in polygon]
    poly_ys = [p[1] for p in polygon]
    space_box = [min(poly_ys), min(poly_xs), max(poly_ys), max(poly_xs)]  # [ymin, xmin, ymax, xmax]

    best_score = 0.0
    best_type = None
    best_overlap = 0.0
    all_detections = []
    vehicles_in_space = []

    for i, (cls, score, box) in enumerate(zip(classes, scores, boxes)):
        if score > 0.15:  # Lower threshold
            class_name = vehicle_classes.get(int(cls), f'class_{int(cls)}')
            is_vehicle = int(cls) in vehicle_classes

            # Box is already normalized [ymin, xmin, ymax, xmax]
            overlap = box_overlap_ratio(space_box, box)

            detection_info = {
                'class': class_name,
                'score': float(score),
                'box': [float(b) for b in box],
                'overlap': float(overlap)
            }

            if is_vehicle:
                all_detections.append(detection_info)
                if overlap > 0.1:
                    vehicles_in_space.append(detection_info)

            # Track best vehicle that overlaps with space
            if is_vehicle and overlap > 0.1 and score > best_score:
                best_score = float(score)
                best_type = class_name
                best_overlap = overlap

    detected = best_score > 0.2 and best_overlap > 0.1

    return {
        'detected': detected,
        'confidence': best_score,
        'vehicle_type': best_type,
        'overlap': best_overlap,
        'all_detections': all_detections[:10],
        'vehicles_in_space': vehicles_in_space
    }


def detect_vehicles_tf(image: np.ndarray, polygon: List[List[float]], debug_info: dict = None) -> Tuple[bool, float, Optional[str]]:
    """Run BOTH OpenCV DNN and TFLite detection for comparison.

    Runs both backends on the full image and checks if any vehicle overlaps with the parking space.
    Both results are stored in debug_info for comparison.

    Primary result comes from OpenCV DNN, TFLite is used for comparison.

    Returns:
        Tuple of (vehicle_detected, confidence, vehicle_type)
    """
    import time

    h, w = image.shape[:2]

    # Convert polygon to normalized bounding box for debug visualization
    poly_xs = [p[0] for p in polygon]
    poly_ys = [p[1] for p in polygon]
    space_box = [min(poly_ys), min(poly_xs), max(poly_ys), max(poly_xs)]

    if debug_info is not None:
        debug_info['dnn_space_box'] = [float(b) for b in space_box]

    # Run OpenCV DNN detection
    opencv_start = time.time()
    opencv_boxes, opencv_classes, opencv_scores = _detect_with_opencv_dnn(image)
    opencv_time = (time.time() - opencv_start) * 1000  # ms

    opencv_result = None
    if len(opencv_boxes) > 0 or _dnn_net is not None:
        opencv_result = _analyze_detections_for_space(
            opencv_boxes, opencv_classes, opencv_scores,
            VOC_VEHICLE_CLASSES, polygon, image.shape
        )
        opencv_result['status'] = 'success'
        opencv_result['time_ms'] = round(opencv_time, 1)
    else:
        opencv_result = {
            'status': 'model_not_loaded' if _dnn_net is None else 'no_detections',
            'detected': False,
            'confidence': 0.0,
            'vehicle_type': None,
            'time_ms': round(opencv_time, 1),
            'all_detections': [],
            'vehicles_in_space': []
        }

    # Run TFLite detection
    tflite_start = time.time()
    tflite_boxes, tflite_classes, tflite_scores = _detect_with_tflite(image)
    tflite_time = (time.time() - tflite_start) * 1000  # ms

    tflite_result = None
    if len(tflite_boxes) > 0 or _tflite_interpreter is not None:
        tflite_result = _analyze_detections_for_space(
            tflite_boxes, tflite_classes, tflite_scores,
            COCO_VEHICLE_CLASSES, polygon, image.shape
        )
        tflite_result['status'] = 'success'
        tflite_result['time_ms'] = round(tflite_time, 1)
    else:
        tflite_result = {
            'status': 'model_not_loaded' if _tflite_interpreter is None else 'no_detections',
            'detected': False,
            'confidence': 0.0,
            'vehicle_type': None,
            'time_ms': round(tflite_time, 1),
            'all_detections': [],
            'vehicles_in_space': []
        }

    # Store both results in debug_info
    if debug_info is not None:
        debug_info['opencv_dnn'] = opencv_result
        debug_info['tflite'] = tflite_result
        debug_info['backend_comparison'] = {
            'opencv_detected': opencv_result.get('detected', False),
            'tflite_detected': tflite_result.get('detected', False),
            'opencv_confidence': opencv_result.get('confidence', 0.0),
            'tflite_confidence': tflite_result.get('confidence', 0.0),
            'opencv_time_ms': opencv_result.get('time_ms', 0),
            'tflite_time_ms': tflite_result.get('time_ms', 0),
            'agreement': opencv_result.get('detected', False) == tflite_result.get('detected', False)
        }

        # Keep legacy fields for compatibility
        debug_info['dnn_status'] = opencv_result.get('status', 'unknown')
        debug_info['dnn_detections'] = opencv_result.get('all_detections', [])
        debug_info['dnn_vehicles_in_space'] = opencv_result.get('vehicles_in_space', [])
        debug_info['dnn_best_vehicle'] = {
            'type': opencv_result.get('vehicle_type'),
            'score': opencv_result.get('confidence', 0.0),
            'overlap': opencv_result.get('overlap', 0.0)
        }
        debug_info['backend'] = 'dual'

    # Primary result from OpenCV DNN (or TFLite if DNN not available)
    if opencv_result.get('status') == 'success':
        return opencv_result.get('detected', False), opencv_result.get('confidence', 0.0), opencv_result.get('vehicle_type')
    elif tflite_result.get('status') == 'success':
        return tflite_result.get('detected', False), tflite_result.get('confidence', 0.0), tflite_result.get('vehicle_type')
    else:
        return False, 0.0, None


def detect_vehicles_tf_OLD(image: np.ndarray, polygon: List[List[float]], debug_info: dict = None) -> Tuple[bool, float, Optional[str]]:
    """DEPRECATED: Old single-backend detection. Kept for reference.
    """
    # Choose backend based on configuration
    if _dnn_backend == 'tflite':
        boxes, classes, scores = _detect_with_tflite(image)
        vehicle_classes = COCO_VEHICLE_CLASSES
        backend_name = 'tflite'
    else:
        net = _load_dnn_model()
        if net is None:
            if debug_info is not None:
                debug_info['dnn_status'] = 'model_not_loaded'
            return False, 0.0, None
        vehicle_classes = VOC_VEHICLE_CLASSES
        backend_name = 'dnn'

    try:
        h, w = image.shape[:2]

        # Convert polygon to normalized bounding box
        poly_xs = [p[0] for p in polygon]
        poly_ys = [p[1] for p in polygon]
        space_box = [min(poly_ys), min(poly_xs), max(poly_ys), max(poly_xs)]  # [ymin, xmin, ymax, xmax]

        if debug_info is not None:
            debug_info['dnn_space_box'] = [float(b) for b in space_box]
            debug_info['backend'] = backend_name

        # Run detection based on backend
        if _dnn_backend == 'dnn':
            # OpenCV DNN detection
            blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # Parse detections - shape is [1, 1, N, 7]
            # Each detection: [batch_id, class_id, confidence, x1, y1, x2, y2]
            boxes = []
            classes = []
            scores = []

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.15:  # Low threshold, filter later
                    class_id = int(detections[0, 0, i, 1])
                    x1 = detections[0, 0, i, 3]
                    y1 = detections[0, 0, i, 4]
                    x2 = detections[0, 0, i, 5]
                    y2 = detections[0, 0, i, 6]
                    # Convert to [ymin, xmin, ymax, xmax] format
                    boxes.append([y1, x1, y2, x2])
                    classes.append(class_id)
                    scores.append(float(confidence))

            boxes = np.array(boxes) if boxes else np.array([])
            classes = np.array(classes) if classes else np.array([])
            scores = np.array(scores) if scores else np.array([])
        # else: TFLite detection already done above

        best_score = 0.0
        best_type = None
        best_overlap = 0.0
        all_detections = []
        vehicles_in_space = []

        # Create debug image - crop to parking space area with padding
        pad = 50  # pixels padding
        space_y1 = max(0, int(space_box[0] * h) - pad)
        space_x1 = max(0, int(space_box[1] * w) - pad)
        space_y2 = min(h, int(space_box[2] * h) + pad)
        space_x2 = min(w, int(space_box[3] * w) + pad)
        debug_image = image[space_y1:space_y2, space_x1:space_x2].copy()
        debug_h, debug_w = debug_image.shape[:2]

        if debug_info is not None:
            debug_info['dnn_crop_size'] = f"{debug_w}x{debug_h}"

        # Draw the parking space boundary on debug image
        for i, p in enumerate(polygon):
            x1 = int(p[0] * w) - space_x1
            y1 = int(p[1] * h) - space_y1
            next_p = polygon[(i + 1) % len(polygon)]
            x2 = int(next_p[0] * w) - space_x1
            y2 = int(next_p[1] * h) - space_y1
            cv2.line(debug_image, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta for space boundary

        for i, (cls, score, box) in enumerate(zip(classes, scores, boxes)):
            if score > 0.15:  # Lower threshold
                class_name = vehicle_classes.get(cls, f'class_{cls}')
                is_vehicle = cls in vehicle_classes

                # Box is already normalized [ymin, xmin, ymax, xmax]
                # Check if this detection overlaps with parking space
                overlap = box_overlap_ratio(space_box, box)

                detection_info = {
                    'class': class_name,
                    'score': float(score),
                    'box': [float(b) for b in box],
                    'overlap': float(overlap)
                }

                if is_vehicle:
                    all_detections.append(detection_info)

                # Draw on debug image if it's near the parking space
                ymin, xmin, ymax, xmax = box
                if boxes_overlap(space_box, box) or overlap > 0:
                    bx1 = int(xmin * w) - space_x1
                    by1 = int(ymin * h) - space_y1
                    bx2 = int(xmax * w) - space_x1
                    by2 = int(ymax * h) - space_y1

                    # Color: green for vehicles in space, yellow for other detections
                    if is_vehicle and overlap > 0.1:
                        color = (0, 255, 0)  # Green - vehicle in space
                        vehicles_in_space.append(detection_info)
                    elif is_vehicle:
                        color = (0, 165, 255)  # Orange - vehicle nearby
                    else:
                        color = (0, 255, 255)  # Yellow - other object

                    cv2.rectangle(debug_image, (bx1, by1), (bx2, by2), color, 2)
                    label = f"{class_name}: {score:.0%}"
                    if overlap > 0:
                        label += f" ({overlap:.0%})"
                    cv2.putText(debug_image, label, (bx1, by1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Track best vehicle that overlaps with space
                if is_vehicle and overlap > 0.1 and score > best_score:
                    best_score = float(score)
                    best_type = class_name
                    best_overlap = overlap

        if debug_info is not None:
            debug_info['dnn_status'] = 'success'
            debug_info['dnn_detections'] = all_detections[:10]
            debug_info['dnn_vehicles_in_space'] = vehicles_in_space
            debug_info['dnn_best_vehicle'] = {'type': best_type, 'score': best_score, 'overlap': best_overlap}
            debug_info['dnn_debug_image'] = encode_image_base64(debug_image)

        # Vehicle detected if good confidence AND overlaps with space
        detected = best_score > 0.2 and best_overlap > 0.1
        return detected, best_score, best_type

    except Exception as e:
        logger.error(f"DNN detection error: {e}")
        if debug_info is not None:
            debug_info['dnn_status'] = f'error: {str(e)}'
        return False, 0.0, None


def detect_space(
    current_image: np.ndarray,
    space: Dict,
    pixel_threshold: float = 0.15,
    dnn_confidence: float = 0.5
) -> Dict:
    """Detect if a parking space is occupied.

    Args:
        current_image: Current camera frame (BGR)
        space: Space definition with polygon and optional reference_image
        pixel_threshold: Threshold for pixel difference (0-1)
        dnn_confidence: Minimum confidence for DNN detection

    Returns:
        Detection result dict with debug info
    """
    polygon = space.get('polygon', [])
    space_id = space.get('id', 0)

    # Initialize debug info
    debug_info = {
        'space_id': space_id,
        'pixel_threshold': pixel_threshold,
        'dnn_confidence_threshold': dnn_confidence,
        'has_reference': bool(space.get('reference_image')),
        'polygon_points': len(polygon)
    }

    if len(polygon) < 3:
        debug_info['error'] = 'Polygon has less than 3 points'
        return {
            'id': space_id,
            'status': 'unknown',
            'confidence': 0.0,
            'method': 'none',
            'vehicle_type': None,
            'debug': debug_info
        }

    # Extract current region
    current_region = extract_region(current_image, polygon, debug_info)

    # Pixel difference detection
    pixel_occupied = False
    pixel_confidence = 0.0
    diff_ratio = 0.0

    reference_b64 = space.get('reference_image')
    if reference_b64:
        try:
            reference_image = decode_base64_image(reference_b64)
            reference_region = extract_region(reference_image, polygon)
            diff_ratio, pixel_confidence = pixel_difference(current_region, reference_region, debug_info)
            pixel_occupied = diff_ratio > pixel_threshold
            debug_info['pixel_occupied'] = pixel_occupied
            debug_info['pixel_method_result'] = 'occupied' if pixel_occupied else 'free'
        except Exception as e:
            logger.warning(f"Pixel diff failed for space {space_id}: {e}")
            debug_info['pixel_error'] = str(e)
    else:
        debug_info['pixel_status'] = 'no_reference_image'

    # DNN detection
    dnn_occupied = False
    dnn_conf = 0.0
    vehicle_type = None

    # Always run DNN detection for comparison
    dnn_occupied, dnn_conf, vehicle_type = detect_vehicles_tf(current_image, polygon, debug_info)
    debug_info['dnn_occupied'] = dnn_occupied
    debug_info['dnn_method_result'] = 'occupied' if dnn_occupied else 'free'

    # Combine results: DNN detection takes priority, pixel diff as backup
    debug_info['decision_logic'] = []

    if dnn_conf > dnn_confidence:
        # DNN has high confidence detection
        final_status = 'occupied'
        final_confidence = dnn_conf
        method = 'dnn'
        debug_info['decision_logic'].append(f'DNN detected vehicle with {dnn_conf:.2f} confidence')
    elif reference_b64 and pixel_confidence > 0:
        # Fall back to pixel difference if we have a reference
        if diff_ratio > pixel_threshold:
            final_status = 'occupied'
            final_confidence = max(pixel_confidence, 0.7)
            method = 'pixel'
            debug_info['decision_logic'].append(f'Pixel diff {diff_ratio:.3f} > threshold {pixel_threshold}')
        else:
            final_status = 'free'
            final_confidence = pixel_confidence
            method = 'pixel'
            debug_info['decision_logic'].append(f'Pixel diff {diff_ratio:.3f} <= threshold {pixel_threshold}')
    else:
        # No reference and no DNN detection - assume free
        final_status = 'free'
        final_confidence = 0.5
        method = 'inference'
        debug_info['decision_logic'].append('No vehicle indicators found')

    debug_info['final_decision'] = {
        'status': final_status,
        'confidence': final_confidence,
        'method': method
    }

    return {
        'id': space_id,
        'status': final_status,
        'confidence': round(final_confidence, 2),
        'method': method,
        'vehicle_type': vehicle_type,
        'debug': debug_info
    }


def detect_all_spaces(
    image_base64: str,
    spaces: List[Dict],
    pixel_threshold: float = 0.15,
    dnn_confidence: float = 0.5,
    reference_features: Dict = None
) -> Tuple[List[Dict], np.ndarray, Dict]:
    """Detect status of all parking spaces.

    Args:
        image_base64: Base64 encoded image
        spaces: List of space definitions
        pixel_threshold: Threshold for pixel difference
        dnn_confidence: Minimum DNN confidence
        reference_features: Optional reference features for image alignment

    Returns:
        Tuple of (results list, annotated image, global_info dict)
    """
    image = decode_base64_image(image_base64)
    results = []
    global_info = {
        'alignment': None,
        'vehicle_boxes': []
    }

    logger.info(f"Processing image {image.shape[1]}x{image.shape[0]} with {len(spaces)} spaces")

    # Try to align image to reference if features available
    aligned_image = image
    if reference_features:
        aligned_image, aligned, alignment_info = align_image_to_reference(
            image, reference_features
        )
        global_info['alignment'] = alignment_info
        if aligned:
            logger.info(f"Image aligned successfully")

    # Run ML detection on full image once to get all vehicle boxes
    all_vehicle_boxes = []
    h, w = aligned_image.shape[:2]
    try:
        if _dnn_backend == 'tflite':
            # TFLite detection
            boxes, classes, scores = _detect_with_tflite(aligned_image)
            vehicle_classes = COCO_VEHICLE_CLASSES
            for box, cls, score in zip(boxes, classes, scores):
                if cls in vehicle_classes and score > 0.4:
                    ymin, xmin, ymax, xmax = box
                    all_vehicle_boxes.append({
                        'type': vehicle_classes[cls],
                        'score': float(score),
                        'box': [float(ymin), float(xmin), float(ymax), float(xmax)],
                        'box_pixels': [
                            int(xmin * w), int(ymin * h),
                            int(xmax * w), int(ymax * h)
                        ]
                    })
        else:
            # OpenCV DNN detection
            net = _load_dnn_model()
            if net is not None:
                vehicle_classes = VOC_VEHICLE_CLASSES
                blob = cv2.dnn.blobFromImage(aligned_image, 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    class_id = int(detections[0, 0, i, 1])

                    if class_id in vehicle_classes and confidence > 0.4:
                        x1 = detections[0, 0, i, 3]
                        y1 = detections[0, 0, i, 4]
                        x2 = detections[0, 0, i, 5]
                        y2 = detections[0, 0, i, 6]

                        all_vehicle_boxes.append({
                            'type': vehicle_classes[class_id],
                            'score': float(confidence),
                            'box': [float(y1), float(x1), float(y2), float(x2)],
                            'box_pixels': [
                                int(x1 * w), int(y1 * h),
                                int(x2 * w), int(y2 * h)
                            ]
                        })
        global_info['vehicle_boxes'] = all_vehicle_boxes
    except Exception as e:
        logger.error(f"Global detection failed: {e}")

    for space in spaces:
        result = detect_space(aligned_image, space, pixel_threshold, dnn_confidence)
        results.append(result)

        # Log result
        logger.info(f"Space {result['id']}: {result['status']} ({result['confidence']:.0%}) via {result['method']}")

        # Draw on image
        polygon = space.get('polygon', [])
        if len(polygon) >= 3:
            h, w = aligned_image.shape[:2]
            points = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon], np.int32)

            color = (0, 255, 0) if result['status'] == 'free' else (0, 0, 255)
            cv2.polylines(aligned_image, [points], True, color, 3)

            # Add label with more info
            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))
            label = f"{result['id']}: {result['status']}"
            conf_label = f"{result['confidence']:.0%} ({result['method']})"

            # Background for text
            cv2.rectangle(aligned_image, (center_x - 50, center_y - 25), (center_x + 50, center_y + 15), (0, 0, 0), -1)
            cv2.putText(aligned_image, label, (center_x - 45, center_y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(aligned_image, conf_label, (center_x - 45, center_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return results, aligned_image, global_info


def generate_privacy_visualization(
    image_shape: Tuple[int, int, int],
    spaces: List[Dict],
    results: List[Dict],
    vehicle_boxes: List[Dict]
) -> np.ndarray:
    """Generate a privacy-preserving visualization with parking spaces and vehicle boxes.

    Instead of showing the actual image, shows a schematic with:
    - Parking space outlines (green=free, red=occupied)
    - Red boxes where vehicles are detected

    Args:
        image_shape: Shape of original image (h, w, c)
        spaces: Space definitions with polygons
        results: Detection results for each space
        vehicle_boxes: List of detected vehicle bounding boxes

    Returns:
        Privacy-preserving visualization image
    """
    h, w = image_shape[:2]

    # Create dark background
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    viz[:] = (30, 30, 30)  # Dark gray background

    # Draw grid pattern for reference
    grid_color = (50, 50, 50)
    for x in range(0, w, 50):
        cv2.line(viz, (x, 0), (x, h), grid_color, 1)
    for y in range(0, h, 50):
        cv2.line(viz, (0, y), (w, y), grid_color, 1)

    # Draw vehicle boxes as red rectangles (privacy-preserving indicator)
    for vbox in vehicle_boxes:
        x1, y1, x2, y2 = vbox['box_pixels']
        # Fill with semi-transparent red
        overlay = viz.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.5, viz, 0.5, 0, viz)
        # Border
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Label
        label = f"{vbox['type']}"
        cv2.putText(viz, label, (x1 + 5, y1 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw parking spaces
    for space, result in zip(spaces, results):
        polygon = space.get('polygon', [])
        if len(polygon) >= 3:
            points = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon], np.int32)

            # Color based on status
            if result['status'] == 'free':
                fill_color = (0, 100, 0)  # Dark green fill
                border_color = (0, 255, 0)  # Bright green border
            else:
                fill_color = (0, 0, 100)  # Dark red fill
                border_color = (0, 0, 255)  # Bright red border

            # Fill polygon
            overlay = viz.copy()
            cv2.fillPoly(overlay, [points], fill_color)
            cv2.addWeighted(overlay, 0.4, viz, 0.6, 0, viz)

            # Border
            cv2.polylines(viz, [points], True, border_color, 2)

            # Label
            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))

            label = f"P{result['id']}"
            status = "FREE" if result['status'] == 'free' else "OCCUPIED"

            cv2.putText(viz, label, (center_x - 15, center_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(viz, status, (center_x - 30, center_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)

    # Add legend
    legend_y = h - 80
    cv2.rectangle(viz, (10, legend_y), (200, h - 10), (40, 40, 40), -1)
    cv2.putText(viz, "Legend:", (15, legend_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(viz, (15, legend_y + 30), (30, legend_y + 45), (0, 255, 0), -1)
    cv2.putText(viz, "Free", (35, legend_y + 42),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(viz, (80, legend_y + 30), (95, legend_y + 45), (0, 0, 255), -1)
    cv2.putText(viz, "Occupied", (100, legend_y + 42),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(viz, (15, legend_y + 50), (30, legend_y + 65), (0, 0, 180), -1)
    cv2.putText(viz, "Vehicle detected", (35, legend_y + 62),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return viz
