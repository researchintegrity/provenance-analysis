# Provenance Analysis Microservice

This directory contains the implementation of the Provenance Analysis Microservice with optimized async processing.

## Architecture

The microservice implements a 5-step pipeline for detecting content sharing between images:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MicroserviceAnalysisRequest                       │
│  - images: List[ImageInfo]     (pool of images to analyze)          │
│  - query_image: ImageInfo      (starting point for analysis)        │
│  - k: int                      (Top-K candidates from CBIR)         │
│  - q: int                      (Top-Q for expansion)                │
│  - max_depth: int              (how deep to expand)                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: IMAGE VISIBILITY CHECK & INDEXING                          │
│  ─────────────────────────────────────────                          │
│  • Check which images are already indexed in CBIR                   │
│  • Batch index any missing images (configurable batch size)         │
│  • Uses user_id for multi-tenant isolation                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: INITIAL CBIR SEARCH (Top-K)                                │
│  ───────────────────────────────────                                │
│  • Query CBIR with the query_image                                  │
│  • Get Top-K most similar images from the pool                      │
│  • Filter results to only include provided images                   │
│  • Create initial matching queue: [(query, cand1), (query, cand2)]  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: PARALLEL DESCRIPTOR EXTRACTION                             │
│  ──────────────────────────────────────                             │
│  • ThreadPoolExecutor with configurable workers                     │
│  • Priority queue: images needed sooner get extracted first         │
│  • Disk caching: descriptors saved as .npy files                    │
│  • Pre-fetching: extract descriptors for upcoming matches           │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: MATCHING PIPELINE (Main Loop)                              │
│  ─────────────────────────────────────                              │
│  For each pair in matching_queue:                                   │
│    1. Wait for descriptors (from cache or extraction)               │
│    2. G2NN keypoint matching                                        │
│    3. MAGSAC++ geometric verification                               │
│    4. Compute shared area (convex hull)                             │
│    5. If match found AND depth < max_depth:                         │
│       → CBIR search Top-Q from matched image                        │
│       → Add new pairs to matching_queue                             │
│       → Submit descriptor extraction for new candidates             │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: GRAPH BUILDING                                             │
│  ──────────────────────────                                         │
│  • Build weighted graph from all matches                            │
│  • Edge weights = shared area percentage                            │
│  • Compute Maximum Spanning Tree (MST)                              │
│  • Find connected components                                        │
│  • Return adjacency matrix for visualization                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Top-K vs Top-Q
- **Top-K** (`k` parameter): Number of candidates from the initial CBIR search
- **Top-Q** (`q` parameter): Number of candidates when expanding from a matched image
- Typically `q < k` since expansion searches are more focused

### Max Depth
Controls how far the algorithm explores from the query image:
- `depth=1`: Only match query vs Top-K candidates
- `depth=2`: Also match candidates vs their Top-Q neighbors
- `depth=3`: Continue one more level of expansion

### Descriptor Caching
Descriptors are cached to disk with filenames like:
```
{output_dir}/{image_id}_{descriptor_type}_kps.npy   # Keypoints
{output_dir}/{image_id}_{descriptor_type}_desc.npy  # Descriptors
{output_dir}/{image_id}_{descriptor_type}_flip_kps.npy   # Flipped keypoints
{output_dir}/{image_id}_{descriptor_type}_flip_desc.npy  # Flipped descriptors
```

This means subsequent runs are much faster as descriptors are reused.

## Usage Examples

### Example 1: Basic Python Usage

```python
from src import ProvenanceMicroservice, RestCBIRClient
from src.schemas import MicroserviceAnalysisRequest, MicroserviceImageInput

# 1. Setup CBIR client
cbir = RestCBIRClient(
    endpoint_url="http://localhost:8001",
    user_id="user_123",
    path_mapping={
        "/local/images": "/container/images"  # Map local to container paths
    }
)

# 2. Verify CBIR is healthy
health = cbir.health_check()
if not health['healthy']:
    raise RuntimeError(f"CBIR not available: {health}")

# 3. Create service
service = ProvenanceMicroservice(
    cbir_client=cbir,
    default_output_dir="./output/descriptors"
)

# 4. Load your images
images = [
    MicroserviceImageInput(id="img_001", path="/local/images/001.png", label="Blot A"),
    MicroserviceImageInput(id="img_002", path="/local/images/002.png", label="Blot B"),
    MicroserviceImageInput(id="img_003", path="/local/images/003.png", label="Blot C"),
    # ... add more images
]

# 5. Create analysis request
request = MicroserviceAnalysisRequest(
    images=images,
    query_image=images[0],  # Start from first image
    k=10,                   # Get 10 candidates from CBIR
    q=5,                    # Expand with 5 candidates on match
    max_depth=3,            # Explore up to depth 3
    descriptor_type="cv_rsift",  # Use RootSIFT descriptors
    max_workers=4,          # 4 parallel extraction workers
    extract_flip=True       # Also check horizontally flipped images
)

# 6. Run analysis
response = service.analyze(request)

# 7. Process results
if response.success:
    print(f"Processing time: {response.processing_time_seconds:.2f}s")
    print(f"Pairs checked: {response.matching_status.total_pairs_checked}")
    print(f"Matches found: {response.matching_status.matched_pairs}")
    
    # Access matched pairs
    for pair in response.matched_pairs:
        print(f"  {pair.image1_id} <-> {pair.image2_id}")
        print(f"    Shared area: {pair.shared_area_img1:.1%} / {pair.shared_area_img2:.1%}")
        print(f"    Keypoints: {pair.matched_keypoints}")
    
    # Access graph
    print(f"\nGraph: {len(response.graph.nodes)} nodes, {len(response.graph.edges)} edges")
    print(f"Spanning tree: {len(response.graph.spanning_tree_edges)} edges")
    print(f"Connected components: {len(response.graph.connected_components)}")
```

### Example 2: Using with REST API

```python
import requests

# API endpoint
API_URL = "http://localhost:8000"

# Check health
health = requests.get(f"{API_URL}/health").json()
print(f"Service healthy: {health['status']}")

# Prepare request
payload = {
    "images": [
        {"id": "img1", "path": "/data/images/001.png", "label": "Sample 1"},
        {"id": "img2", "path": "/data/images/002.png", "label": "Sample 2"},
        {"id": "img3", "path": "/data/images/003.png", "label": "Sample 3"},
    ],
    "query_image": {"id": "img1", "path": "/data/images/001.png"},
    "k": 10,
    "q": 5,
    "max_depth": 2,
    "descriptor_type": "cv_rsift",
    "max_workers": 4
}

# Run analysis
response = requests.post(f"{API_URL}/analyze", json=payload)
result = response.json()

if result["success"]:
    print(f"Found {result['matching_status']['matched_pairs']} matches")
    for pair in result["matched_pairs"]:
        print(f"  {pair['image1_id']} <-> {pair['image2_id']}")
```

### Example 3: Visualizing Results with NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt

# After running analysis...
response = service.analyze(request)

# Build NetworkX graph from spanning tree
G = nx.Graph()

# Add nodes
for node in response.graph.nodes:
    G.add_node(node.id, label=node.label, is_query=node.is_query)

# Add spanning tree edges only (cleaner visualization)
for edge in response.graph.spanning_tree_edges:
    G.add_edge(
        edge.source, 
        edge.target,
        weight=edge.shared_area_source,
        keypoints=edge.matched_keypoints
    )

# Visualize
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=2)

# Color query node differently
colors = ['red' if G.nodes[n].get('is_query') else 'lightblue' for n in G.nodes()]
nx.draw(G, pos, node_color=colors, node_size=800, with_labels=True, font_size=8)

# Add edge labels
edge_labels = {(u, v): f"{G[u][v]['keypoints']} kpts" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

plt.title("Provenance Graph (Spanning Tree)")
plt.show()
```

## Key Features

### Performance Optimizations
- **Parallel Descriptor Extraction**: Uses ThreadPoolExecutor for concurrent extraction
- **Priority-Based Processing**: Higher priority for images needed sooner
- **Disk Caching**: Descriptors are cached to disk and reused across runs
- **Pre-fetching**: Descriptors for upcoming matches are extracted in advance
- **Batch CBIR Indexing**: Missing images are indexed in configurable batches
- **Deduplication**: CBIR results are deduplicated to handle duplicate index entries

### CBIR Integration
- Abstract `CBIRClient` interface for flexibility
- `RestCBIRClient` for production CBIR systems with Milvus backend
- `MockCBIRClient` for testing without a real CBIR system
- Automatic visibility checking and indexing
- User-based isolation (multi-tenant support)

## Files

- `src/api.py`: FastAPI application with REST endpoints
- `src/service.py`: Core `ProvenanceMicroservice` class
- `src/descriptor_manager.py`: Async descriptor extraction manager
- `src/cbir.py`: CBIR client implementations
- `src/schemas.py`: Pydantic models for API
- `src/matcher.py`: Geometric matching logic
- `src/graph_builder.py`: Provenance graph construction
- `src/keypoint_extractor.py`: Descriptor extraction functions

## Setup

1. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Service

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVENANCE_OUTPUT_DIR` | `/tmp/provenance_output` | Directory for descriptor cache |
| `CBIR_ENDPOINT` | `http://localhost:8001` | CBIR service URL |
| `CBIR_USER_ID` | `provenance_service` | User ID for CBIR |
| `USE_MOCK_CBIR` | `false` | Use mock CBIR client |
| `LOCAL_DATA_PATH` | - | Local path prefix for path mapping |
| `REMOTE_DATA_PATH` | - | Remote path prefix for path mapping |

### Start the Server

```bash
# With real CBIR
CBIR_ENDPOINT=http://cbir-service:8001 uvicorn src.api:app --host 0.0.0.0 --port 8000

# With mock CBIR (for testing)
USE_MOCK_CBIR=true uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Documentation is available at `http://localhost:8000/docs`.

## API Endpoints

### POST /analyze

Main analysis endpoint for provenance detection.

**Request Body:**
```json
{
    "images": [
        {"id": "img1", "path": "/data/image1.jpg", "label": "Western Blot"},
        {"id": "img2", "path": "/data/image2.jpg", "label": "Western Blot"}
    ],
    "query_image": {"id": "query", "path": "/data/query.jpg"},
    "k": 10,
    "q": 5,
    "max_depth": 3,
    "descriptor_type": "cv_rsift",
    "max_workers": 4,
    "cbir_batch_size": 32
}
```

**Response:**
```json
{
    "success": true,
    "message": "Analysis completed successfully",
    "processing_time_seconds": 12.5,
    "indexing_status": {
        "total_images": 100,
        "already_indexed": 95,
        "newly_indexed": 5,
        "failed_to_index": 0
    },
    "extraction_status": {
        "total_images": 15,
        "extracted": 15,
        "from_cache": 5,
        "failed": 0
    },
    "matching_status": {
        "total_pairs_checked": 25,
        "matched_pairs": 8,
        "expansion_count": 3
    },
    "graph": {
        "nodes": [...],
        "edges": [...],
        "spanning_tree_edges": [...],
        "connected_components": [...]
    },
    "matched_pairs": [...]
}
```

### POST /index

Pre-index images in CBIR system.

### POST /visibility

Check which images are visible in CBIR.

### GET /health

Health check with CBIR connectivity status.

### GET /config

Get current service configuration.

## Docker Compose (Recommended)

The easiest way to run the full stack is with Docker Compose:

### Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env to set DATA_PATH to your image directory

# 2. Start all services
docker-compose up -d

# 3. Check status
docker-compose ps
docker-compose logs -f provenance-service

# 4. Test the services
curl http://localhost:8000/health  # Provenance API
curl http://localhost:8001/health  # CBIR API
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `provenance-service` | 8000 | Provenance analysis API |
| `cbir-service` | 8001 | CBIR (image search) API |
| `milvus` | 19530 | Vector database |
| `minio` | 9000/9001 | Object storage |
| `etcd` | 2379 | Distributed KV store |
| `attu` | 3322 | Milvus admin UI (optional) |

### Environment Variables

Create a `.env` file with:

```bash
# Path to your images on the host
DATA_PATH=/path/to/your/images

# Path mapping (if your app uses different paths)
LOCAL_DATA_PATH=/home/user/data
# Container sees files at /workspace

# CBIR user isolation
CBIR_USER_ID=my_project

# Logging level
LOG_LEVEL=INFO

# GPU support (optional)
MODEL_DEVICE=cpu  # or cuda
```

### GPU Support

For GPU acceleration:

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Common Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f provenance-service
docker-compose logs -f cbir-service

# Restart a service
docker-compose restart provenance-service

# Remove all data (reset)
docker-compose down -v

# Start with admin UI
docker-compose --profile admin up -d
```

### Path Mapping

When your host paths differ from container paths, use path mapping:

```bash
# In .env
LOCAL_DATA_PATH=/home/user/my-images
# Container mount: DATA_PATH → /workspace

# Your API requests use host paths:
# { "path": "/home/user/my-images/img.png" }
# Internally mapped to: /workspace/img.png
```

## Docker (Standalone)

Build and run without docker-compose:

```bash
# Build
docker build -f Dockerfile.api -t provenance-analysis .

# Run with real CBIR
docker run -p 8000:8000 \
    -e CBIR_ENDPOINT=http://cbir-service:8001 \
    -v /data:/data \
    provenance-analysis

# Run with mock CBIR
docker run -p 8000:8000 \
    -e USE_MOCK_CBIR=true \
    -v /data:/data \
    provenance-analysis
```

## Python Usage

```python
from src import ProvenanceMicroservice, MockCBIRClient
from src.schemas import MicroserviceAnalysisRequest, MicroserviceImageInput

# Setup
cbir_client = MockCBIRClient()
service = ProvenanceMicroservice(cbir_client, output_dir="/tmp/provenance")

# Build request
request = MicroserviceAnalysisRequest(
    images=[
        MicroserviceImageInput(id="img1", path="/path/to/img1.jpg"),
        MicroserviceImageInput(id="img2", path="/path/to/img2.jpg"),
    ],
    query_image=MicroserviceImageInput(id="query", path="/path/to/query.jpg"),
    k=10,
    q=5,
    max_depth=3
)

# Run analysis
response = service.analyze(request)

# Access results
print(f"Found {len(response.matched_pairs)} matches")
print(f"Graph has {len(response.graph.nodes)} nodes")
```

## Running Tests

```bash
# Run test script
python notebooks/test_microservice.py
```
