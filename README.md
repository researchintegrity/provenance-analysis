# Provenance Analysis Module

Content sharing detection for image provenance analysis in the ELIS system.

## Overview

This module detects shared content between scientific images using computer vision techniques. It's designed to identify image manipulation, duplication, and reuse across scientific publications.

### Core Pipeline

1. **CBIR Integration**: Uses a Content-Based Image Retrieval system to find candidate similar images
2. **Keypoint Extraction**: SIFT-based descriptors (VLFeat with histogram equalization or OpenCV RootSIFT)
3. **Keypoint Matching**: G2NN (Generalized 2-Nearest Neighbor) for robust match selection
4. **Geometric Verification**: MAGSAC++ for filtering false matches via homography estimation
5. **Shared Area Computation**: Convex hull-based calculation of overlapping content
6. **Graph Building**: Constructs a provenance graph showing content relationships

### Algorithm Flow

```
Query Image
    │
    ▼
┌─────────────────┐
│  CBIR Search    │──► Top-K similar images
│   (Top-K)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extract SIFT   │──► Keypoints + Descriptors
│  Descriptors    │    (cached to disk)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  G2NN Matching  │──► Candidate matches
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MAGSAC++       │──► Geometrically verified
│  Verification   │    inlier matches
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Match Found?   │─Yes─►  CBIR Search    │──► Expand with Top-Q
└────────┬────────┘     │   (Top-Q)       │    (recursive)
         │              └─────────────────┘
         No
         │
         ▼
┌─────────────────┐
│  Build Graph    │──► Provenance Graph
│  (MST + CC)     │    + Spanning Tree
└─────────────────┘
```

## Quick Start with Docker Compose

The easiest way to run the provenance analysis is with Docker Compose. This service requires the **CBIR system** to be running first.

### 1. Prerequisites

- Docker & Docker Compose
- **CBIR system running** (see `../cbir-system/README.md`)

### 2. Start CBIR System First

```bash
# In the cbir-system directory
cd ../cbir-system
cp .env.example .env
# Edit .env to set WORKSPACE_PATH to your image directory
docker-compose up -d

# Verify CBIR is running
curl http://localhost:8001/health
```

### 3. Start Provenance Service

```bash
# In the provenance-analysis directory
cd ../provenance-analysis
cp .env.example .env
# Edit .env to configure paths
docker-compose up -d

# Verify service is running
curl http://localhost:8002/health
```

### 4. Environment Configuration

Edit `.env` to configure the service:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVENANCE_PORT` | `8002` | Provenance service port |
| `CBIR_ENDPOINT` | `http://host.docker.internal:8001` | CBIR service URL |
| `CBIR_USER_ID` | `provenance_service` | User ID for CBIR isolation |
| `DATA_PATH` | `./data` | Host path to images (mounted as `/workspace`) |
| `LOCAL_DATA_PATH` | - | Host path prefix for path mapping |
| `LOG_LEVEL` | `INFO` | Logging level |

**Important**: `LOCAL_DATA_PATH` must match the path prefix you use in API requests. The service maps these to `/workspace` inside the container.

### 5. Test the Service

```bash
# Create a test request
cat > /tmp/test_request.json << 'EOF'
{
    "images": [
        {"id": "img1", "path": "/your/host/path/image1.png", "label": "test"},
        {"id": "img2", "path": "/your/host/path/image2.png", "label": "test"}
    ],
    "query_image": {"id": "img1", "path": "/your/host/path/image1.png"},
    "k": 10,
    "q": 5,
    "max_depth": 2
}
EOF

# Run analysis
curl -X POST http://localhost:8002/analyze \
  -H "Content-Type: application/json" \
  -d @/tmp/test_request.json
```

### 6. Common Commands

```bash
# Start the service
docker-compose up -d

# Stop the service
docker-compose down

# View logs
docker-compose logs -f provenance-service

# Restart after code changes
docker restart provenance-service
```

---

## Using as a Python Library

See [README_MICROSERVICE.md](README_MICROSERVICE.md) for full microservice documentation.

```python
from src import ProvenanceMicroservice, RestCBIRClient
from src.schemas import MicroserviceAnalysisRequest, MicroserviceImageInput

# Connect to CBIR service
cbir = RestCBIRClient(
    endpoint_url="http://localhost:8001",
    user_id="my_user",
    path_mapping={"/local/path": "/container/path"}  # Optional
)

# Create service
service = ProvenanceMicroservice(cbir, output_dir="./descriptors")

# Prepare images
images = [
    MicroserviceImageInput(id="img1", path="/path/to/img1.png", label="Western Blot"),
    MicroserviceImageInput(id="img2", path="/path/to/img2.png", label="Western Blot"),
    # ... more images
]

# Run analysis
request = MicroserviceAnalysisRequest(
    images=images,
    query_image=images[0],  # Query image
    k=10,                   # Top-K candidates from CBIR
    q=5,                    # Top-Q for expansion on match
    max_depth=3,            # Maximum expansion depth
    max_workers=4           # Parallel descriptor extraction
)

response = service.analyze(request)

# Results
print(f"Matches: {response.matching_status.matched_pairs}")
print(f"Nodes: {len(response.graph.nodes)}")
print(f"Edges: {len(response.graph.edges)}")
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Run provenance analysis |
| `/index` | POST | Pre-index images in CBIR |
| `/visibility` | POST | Check which images are indexed |
| `/health` | GET | Health check with CBIR status |
| `/config` | GET | Get current configuration |

See `http://localhost:8002/docs` for full API documentation.

---

## Docker CLI Usage

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
