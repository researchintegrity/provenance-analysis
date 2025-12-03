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

## 🚀 Quick Start with Docker

The easiest way to get started is using Docker Compose. This automatically handles all dependencies and networking.

### Prerequisites

- **Docker & Docker Compose** installed
- **CBIR system running** (see `../cbir-system/README.md`)

### Step 1: Start the CBIR System

```bash
# Navigate to CBIR system directory
cd ../cbir-system

# Copy and configure environment
cp .env.example .env
# Edit .env to set your workspace path (where your images are stored)

# Start CBIR services
docker-compose up -d

# Verify CBIR is running
curl http://localhost:8001/health
# Should return: {"status":"healthy","model":true,"database":true}
```

### Step 2: Start Provenance Analysis

```bash
# Navigate to provenance analysis directory
cd ../provenance-analysis

# Copy and configure environment
cp .env.example .env
# Edit .env to set DATA_PATH to your image directory

# Start the service
docker-compose up -d

# Verify service is running
curl http://localhost:8002/health
# Should return: {"status":"healthy","version":"2.0.0","cbir_connected":true}
```

### Step 3: Test with Sample Images

Here's a complete working example. Replace `/path/to/your/images` with your actual image directory:

```bash
# Example API request
curl -X POST http://localhost:8002/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "my_user",
    "images": [
      {"id": "img1", "path": "/workspace/cbir-test/00025.png", "label": "test"},
      {"id": "img2", "path": "/workspace/cbir-test/00068.png", "label": "test"},
      {"id": "img3", "path": "/workspace/cbir-test/00084.png", "label": "test"}
    ],
    "query_image": {"id": "query", "path": "/workspace/cbir-test/00025.png", "label": "query"},
    "k": 3,
    "q": 2,
    "max_depth": 2,
    "descriptor_type": "cv_rsift",
    "extract_flip": true,
    "alignment_strategy": "CV_MAGSAC",
    "matching_method": "BF",
    "min_keypoints": 20,
    "min_area": 0.01,
    "max_workers": 2,
    "cbir_batch_size": 32
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Analysis completed successfully",
  "processing_time_seconds": 4.7,
  "indexing_status": {
    "total_images": 4,
    "already_indexed": 3,
    "newly_indexed": 1,
    "failed_to_index": 0
  },
  "extraction_status": {
    "total_images": 3,
    "extracted": 3,
    "from_cache": 0,
    "failed": 0
  },
  "matching_status": {
    "total_pairs_checked": 3,
    "matched_pairs": 2,
    "expansion_count": 1
  },
  "graph": {
    "nodes": [...],
    "edges": [...],
    "connected_components": [...]
  }
}
```

### 📝 API Request Format

The main `/analyze` endpoint accepts this JSON structure:

```json
{
  "user_id": "your_user_id",
  "images": [
    {
      "id": "unique_image_id",
      "path": "/workspace/path/to/image.png",
      "label": "optional_category"
    }
  ],
  "query_image": {
    "id": "query_id", 
    "path": "/workspace/path/to/query.png",
    "label": "query"
  },
  "k": 10,
  "q": 5,
  "max_depth": 3,
  "descriptor_type": "cv_rsift",
  "extract_flip": true,
  "alignment_strategy": "CV_MAGSAC",
  "matching_method": "BF",
  "min_keypoints": 20,
  "min_area": 0.01,
  "max_workers": 4,
  "cbir_batch_size": 32
}
```

**Parameter Explanations:**
- `user_id`: **Required.** User ID for CBIR multi-tenant isolation. Each user has their own image index.
- `images`: Array of images available for analysis
- `query_image`: The image to analyze provenance for
- `k`: Number of top candidates from initial CBIR search (1-100)
- `q`: Number of candidates for expansion searches (1-50)
- `max_depth`: Maximum expansion depth (1-10)
- `descriptor_type`: `"vlfeat_sift_heq"`, `"cv_sift"`, or `"cv_rsift"`
- `extract_flip`: Also analyze horizontally flipped images
- `alignment_strategy`: `"CV_MAGSAC"` (recommended), `"CV_RANSAC"`, `"CV_LMEDS"`
- `matching_method`: `"BF"` (Brute Force) or `"FLANN"`
- `min_keypoints`: Minimum matching keypoints required (4-1000)
- `min_area`: Minimum shared area threshold (0.0-1.0)
- `max_workers`: Parallel processing workers (1-16)
- `cbir_batch_size`: Batch size for CBIR indexing (1-128)

### Environment Configuration

Create a `.env` file with these settings:

```bash
# Service Configuration
PROVENANCE_PORT=8002
LOG_LEVEL=INFO

# CBIR Connection
CBIR_ENDPOINT=http://host.docker.internal:8001

# Data Paths (IMPORTANT!)
DATA_PATH=/path/to/your/image/directory
LOCAL_DATA_PATH=/path/to/your/image/directory
REMOTE_DATA_PATH=/workspace

# Performance
MAX_WORKERS=4
CBIR_BATCH_SIZE=32
```

**Note:** The `user_id` is now passed in each API request, not as an environment variable. This allows multi-tenant usage where different users can have isolated image indexes.

**Path Mapping Explanation:**
- `DATA_PATH`: Host directory containing your images (mounted as `/workspace` in container)
- `LOCAL_DATA_PATH`: Same as DATA_PATH (used for path translation)
- `REMOTE_DATA_PATH`: Always `/workspace` (container internal path)

### 📋 Common Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f provenance-service

# Restart after configuration changes
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build

# Clean up (removes volumes)
docker-compose down -v
```

### 🔧 Troubleshooting

**Service won't start:**
```bash
# Check if CBIR is running first
curl http://localhost:8001/health

# Check provenance logs
docker-compose logs provenance-service

# Restart with fresh build
docker-compose down
docker-compose up -d --build
```

**"No candidates found" error:**
- Ensure CBIR system is running and healthy
- Check that image paths in requests match your `DATA_PATH`
- Verify images exist and are readable

**Path mapping issues:**
- Ensure `LOCAL_DATA_PATH` matches the path prefix in your API requests
- Use `/workspace/` prefix in API calls (container path)
- Check volume mounts: `docker inspect provenance-service`

**Performance issues:**
- Increase `MAX_WORKERS` for faster processing
- Reduce `CBIR_BATCH_SIZE` if memory constrained
- Monitor logs for bottlenecks

### 💡 Quick Test Script

Save this as `test_provenance.sh` and run it:

```bash
#!/bin/bash
# Quick test script for provenance analysis

# Configuration - EDIT THESE PATHS
IMAGE_DIR="/path/to/your/images"  # Your image directory
CBIR_PORT=8001
PROVENANCE_PORT=8002

echo "Testing Provenance Analysis API..."
echo "=================================="

# Check CBIR health
echo "1. Checking CBIR service..."
if curl -s http://localhost:$CBIR_PORT/health | grep -q "healthy"; then
    echo "✅ CBIR is healthy"
else
    echo "❌ CBIR is not responding. Start CBIR first:"
    echo "   cd ../cbir-system && docker-compose up -d"
    exit 1
fi

# Check Provenance health
echo "2. Checking Provenance service..."
if curl -s http://localhost:$PROVENANCE_PORT/health | grep -q "healthy"; then
    echo "✅ Provenance service is healthy"
else
    echo "❌ Provenance service is not responding. Start it first:"
    echo "   docker-compose up -d"
    exit 1
fi

# Find some test images
echo "3. Finding test images..."
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Image directory not found: $IMAGE_DIR"
    echo "   Please edit IMAGE_DIR in this script"
    exit 1
fi

# Get first 3 PNG files
IMAGES=($(find "$IMAGE_DIR" -name "*.png" | head -3))
if [ ${#IMAGES[@]} -lt 3 ]; then
    echo "❌ Need at least 3 PNG images in $IMAGE_DIR"
    exit 1
fi

echo "✅ Found ${#IMAGES[@]} test images"

# Create JSON request
JSON=$(cat <<EOF
{
  "images": [
    {"id": "img1", "path": "/workspace/$(basename "${IMAGES[0]}")", "label": "test"},
    {"id": "img2", "path": "/workspace/$(basename "${IMAGES[1]}")", "label": "test"},
    {"id": "img3", "path": "/workspace/$(basename "${IMAGES[2]}")", "label": "test"}
  ],
  "query_image": {"id": "query", "path": "/workspace/$(basename "${IMAGES[0]}")", "label": "query"},
  "k": 3,
  "q": 2,
  "max_depth": 2,
  "descriptor_type": "cv_rsift",
  "extract_flip": true,
  "alignment_strategy": "CV_MAGSAC",
  "matching_method": "BF",
  "min_keypoints": 20,
  "min_area": 0.01,
  "max_workers": 2,
  "cbir_batch_size": 32
}
EOF
)

echo "4. Running provenance analysis..."
echo "   Query image: $(basename "${IMAGES[0]}")"
echo "   Processing..."

# Run the analysis
RESPONSE=$(curl -s -X POST http://localhost:$PROVENANCE_PORT/analyze \
  -H "Content-Type: application/json" \
  -d "$JSON")

# Check if successful
if echo "$RESPONSE" | grep -q '"success":true'; then
    echo "✅ Analysis completed successfully!"
    echo "   Processing time: $(echo "$RESPONSE" | grep -o '"processing_time_seconds":[0-9.]*' | cut -d: -f2)s"
    echo "   Matched pairs: $(echo "$RESPONSE" | grep -o '"matched_pairs":[0-9]*' | cut -d: -f2)"
    echo ""
    echo "Full response saved to: test_response.json"
    echo "$RESPONSE" > test_response.json
else
    echo "❌ Analysis failed:"
    echo "$RESPONSE"
fi
```

Make it executable and run:

```bash
chmod +x test_provenance.sh
./test_provenance.sh
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
