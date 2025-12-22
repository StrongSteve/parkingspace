# ADR 002: Detection Strategy

## Status
Accepted

## Context
We need to detect whether parking spaces are occupied or free. Options considered:
1. Pixel difference only
2. TensorFlow/ML object detection only
3. Hybrid approach (pixel diff + TensorFlow)

## Decision
Use a hybrid approach: pixel difference as primary method, TensorFlow vehicle detection as secondary validation.

## Rationale

### Pixel Difference
- Fast (~10ms per space)
- Works well for static camera with consistent lighting
- Compares current frame region against calibrated "empty" reference
- Low resource usage

### TensorFlow Vehicle Detection
- Uses COCO-SSD model to detect cars, trucks, motorcycles
- More robust to lighting changes and shadows
- Can identify vehicle type
- Slower (~200-500ms per frame)

### Hybrid Logic
```
1. Run pixel difference on each space
2. If confidence < 0.7, run TensorFlow on that region
3. TensorFlow result overrides pixel diff if confidence is higher
4. Final status = highest confidence method
```

## Consequences

### Positive
- Best of both worlds: speed + accuracy
- Pixel diff handles obvious cases quickly
- TensorFlow catches edge cases
- Fallback if one method fails

### Negative
- More complex codebase
- Higher resource usage when TensorFlow runs
- TensorFlow model download (~5MB) on startup

## Processing Location
Detection runs on the **backend server**, not the mobile device:
- More consistent performance
- No browser throttling issues
- Easier to debug and update
- iPhone just captures and uploads images
