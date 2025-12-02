# Provenance Analysis Module

Content sharing detection for image provenance analysis in the ELIS system.

## Overview

This module detects shared content between images using:
1. **Keypoint extraction**: SIFT-based descriptors (VLFeat with histogram equalization by default)
2. **Keypoint matching**: G2NN (Generalized 2-Nearest Neighbor) selection
3. **Geometric verification**: MAGSAC++ for robust alignment
4. **Shared area computation**: Convex hull-based area calculation

## Docker Usage

### Build the Image

```bash
docker build -t provenance-analysis:latest .
```

### Commands

#### 1. Extract Descriptors

```bash
docker run --rm \
  -v /path/to/images:/data \
  -v /path/to/output:/output \
  provenance-analysis:latest \
  extract -i /data/image.png -o /output -d vlfeat_sift_heq
```

#### 2. Match Two Images

Create a JSON input file:

```json
{
  "image1": {
    "image_path": "/data/image1.png",
    "keypoints_path": "/output/image1_vlfeat_sift_heq_kps.npy",
    "descriptors_path": "/output/image1_vlfeat_sift_heq_desc.npy",
    "flip_keypoints_path": "/output/image1_vlfeat_sift_heq_flip_kps.npy",
    "flip_descriptors_path": "/output/image1_vlfeat_sift_heq_flip_desc.npy"
  },
  "image2": {
    "image_path": "/data/image2.png",
    "keypoints_path": "/output/image2_vlfeat_sift_heq_kps.npy",
    "descriptors_path": "/output/image2_vlfeat_sift_heq_desc.npy"
  },
  "alignment_strategy": "CV_MAGSAC",
  "matching_method": "BF",
  "min_keypoints": 20,
  "min_area": 0.01,
  "check_flip": true
}
```

Run matching:

```bash
docker run --rm \
  -v /path/to/images:/data \
  -v /path/to/output:/output \
  provenance-analysis:latest \
  match -i /output/match_request.json
```

#### 3. Batch Match

Create a batch request JSON:

```json
{
  "pairs": [
    {
      "image1": {...},
      "image2": {...}
    },
    {
      "image1": {...},
      "image2": {...}
    }
  ],
  "parallel": true,
  "max_workers": 4
}
```

Run batch matching:

```bash
docker run --rm \
  -v /path/to/images:/data \
  -v /path/to/output:/output \
  provenance-analysis:latest \
  batch -i /output/batch_request.json
```

#### 4. Generic Run Command (for ELIS integration)

```bash
docker run --rm \
  -v /path/to/data:/data \
  provenance-analysis:latest \
  run -i '{"command": "extract", "extract_request": {...}}'
```

## Output Schemas

### Extract Response

```json
{
  "success": true,
  "command": "extract",
  "message": "Descriptors extracted successfully",
  "extract_response": {
    "success": true,
    "message": "Descriptors extracted successfully",
    "keypoints_path": "/output/image_vlfeat_sift_heq_kps.npy",
    "descriptors_path": "/output/image_vlfeat_sift_heq_desc.npy",
    "flip_keypoints_path": "/output/image_vlfeat_sift_heq_flip_kps.npy",
    "flip_descriptors_path": "/output/image_vlfeat_sift_heq_flip_desc.npy",
    "keypoint_count": 1523,
    "flip_keypoint_count": 1498
  }
}
```

### Match Response

```json
{
  "success": true,
  "command": "match",
  "message": "Matching completed",
  "match_response": {
    "success": true,
    "message": "Matching completed",
    "result": {
      "shared_area_img1": 0.15,
      "shared_area_img2": 0.18,
      "matched_keypoints": 45,
      "is_flipped_match": false,
      "visualization_path": null
    }
  }
}
```

## Supported Descriptor Types

| Type | Description |
|------|-------------|
| `vlfeat_sift_heq` | VLFeat SIFT with histogram equalization (recommended) |
| `cv_sift` | OpenCV SIFT |
| `cv_rsift` | OpenCV RootSIFT |

## Supported Alignment Strategies

| Strategy | Description |
|----------|-------------|
| `CV_MAGSAC` | MAGSAC++ (recommended, most robust) |
| `CV_RANSAC` | Classic RANSAC |
| `CV_LMEDS` | Least Median of Squares |

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_keypoints` | 20 | Minimum matching keypoints required |
| `min_area` | 0.01 | Minimum shared area threshold (1%) |
| `check_flip` | true | Check horizontally flipped images |
| `matching_method` | BF | Keypoint matching (BF or FLANN) |
